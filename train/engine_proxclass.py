import torch
from tqdm import tqdm
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import os
import sys
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image
import numpy as np
from . import metrics
import torch.nn as nn



def train(model, optimizer, loss_fn, train_dl, val_dl, mu=0.1, cid=None, round=None,
          global_params=None, log=None, plot=None, epochs=200, device="cuda"):

    all_logits = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_logits = []
        total = 0
        correct = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            image = batch["image"].to(device, dtype=torch.float32)

            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                target = batch["target"].to(device, dtype=torch.float32).view(-1, 1)
            else:
                target = batch["target"].to(device, dtype=torch.long).view(-1)

            optimizer.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, target)

            prox_reg = 0.0
            for name, param in model.named_parameters():
                w_global = global_params[name].to(device)
                prox_reg += torch.sum((param - w_global)**2)
            loss = loss + (mu/2) * prox_reg

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    epoch_logits.append(logits.view(-1).detach().cpu())
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    correct += (preds.view(-1) == target.view(-1).long()).sum().item()
                    total += target.numel()
                else:
                    epoch_logits.append(logits.detach().cpu())
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)

        epoch_loss /= len(train_dl)
        epoch_acc = correct / total

        epoch_logits = torch.cat(epoch_logits, dim=0)
        avg_epoch_logits = torch.mean(epoch_logits, dim=0)
        all_logits.append(avg_epoch_logits)

        log.log_train_class(epoch, epoch_acc, epoch_loss, cid, round)

        plot.training_loss[cid].append(epoch_loss)
        plot.training_acc[cid].append(epoch_acc)

        val_loss, val_acc, val_auc, _, _, val_logits = evaluate(model, loss_fn, val_dl, device)

        log.log_validate_class(epoch, val_acc, val_auc, val_loss, cid, round)

        plot.val_client_loss[cid].append(val_loss)
        plot.val_client_acc[cid].append(val_acc)
        plot.val_client_auc[cid].append(val_auc)
        plot.client_validation_trajectory[round][cid] = val_logits
    return model, all_logits


@torch.no_grad()
def evaluate(model, loss_fn, dataloader, device, log=None,
             eval=False, round=None, plot=None):

    test_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_targets = []
    model.eval()

    for batch in tqdm(dataloader, desc="Testing", leave=False):
        image = batch["image"].to(device, dtype=torch.float32)

        # BCE vs CCE target handling
        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            target = batch["target"].to(device, dtype=torch.float32).view(-1, 1)
        else:
            target = batch["target"].to(device, dtype=torch.long).view(-1)

        logits = model(image)
        loss = loss_fn(logits, target)
        test_loss += loss.item()

        # Predictions
        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            all_logits.append(logits.view(-1).cpu())
            all_targets.append(target.view(-1).cpu())

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.view(-1) == target.view(-1).long()).sum().item()
            total += target.numel()
        else:
            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())

            preds = torch.argmax(logits, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    test_loss /= len(dataloader)
    acc = correct / total

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        probs = torch.sigmoid(all_logits).cpu().numpy()
        roc_auc = roc_auc_score(all_targets.cpu().numpy(), probs)
        pr, re, _ = precision_recall_curve(all_targets, probs)
    else:
        probs = torch.softmax(all_logits, dim=1).cpu().numpy()
        roc_auc = roc_auc_score(
            all_targets.cpu().numpy(),
            probs,
            multi_class="ovr",
            average="macro"
        )

        num_classes = probs.shape[1]
        targets_bin = label_binarize(
            all_targets.cpu().numpy(),
            classes=np.arange(num_classes)
        )
        pr, re, _ = precision_recall_curve(
            targets_bin.ravel(),
            probs.ravel()
        )
    pr_auc = auc(re, pr)
    f1 = 2 * (pr * re) / (pr + re + 1e-12)
    best_f1 = f1.max()
    best_idx = f1.argmax()
    best_pr = pr[best_idx]
    best_re = re[best_idx]

    if log is not None: log.log_test_class(acc, roc_auc, pr_auc, best_pr, best_re, best_f1, test_loss, round)
    if plot is not None:
        plot.testing_loss.append(test_loss)
        plot.testing_acc.append(acc)
        plot.testing_auc.append(roc_auc)

    return test_loss, acc, roc_auc, pr, re, all_logits
