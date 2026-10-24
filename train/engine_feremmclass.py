import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
import os
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np
from feremm_utils import register_activations, clear_hooks, average_gradients, retrieve_gradients
from sklearn.preprocessing import label_binarize


def train(model, train_dl, optimizer, loss_fn, log, device, epochs=200,
          cid=None, round=None, struct=None):

    all_logits = []
    model.train()
    for ep in range(epochs):
        epoch_logits = []
        epoch_loss = 0.0
        total = 0
        correct = 0
        for batch in tqdm(train_dl, desc=f"Epoch {ep + 1}/{epochs}", leave=False):
            image = batch["image"].to(device, dtype=torch.float32)

            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                target = batch["target"].to(device, dtype=torch.float32).view(-1, 1)
            else:
                target = batch["target"].to(device, dtype=torch.long).view(-1)

            optimizer.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, target)

            loss.backward()
            optimizer.step()

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
            epoch_loss += loss.item()

        epoch_logits = torch.cat(epoch_logits, dim=0)
        avg_epoch_logits = epoch_logits.mean(dim=0)
        all_logits.append(avg_epoch_logits)

        epoch_loss /= len(train_dl)
        epoch_acc = correct / total

        log.log_train_class(ep, cid, epoch_acc, epoch_loss, round)
    return model, all_logits


def validate(model, dataloader, optimizer, loss_fn, device, cid, log=None, round=None,
             struct=None, hook=None, layer=None):

    all_logits = []
    all_targets = []
    val_loss = 0.0
    total = 0
    correct = 0
    model.eval()
    gradients = {l: None for l in layer}
    batch_count = 0

    struct.registered_hooks[cid] = register_activations(model, cid, layer, struct.registered_hooks[cid],
                                                        a_struct=struct.act_tracker, type=hook)
    for batch in tqdm(dataloader, desc=f"VALIDATION Client {cid}: ", leave=False):
        image = batch["image"].to(device, dtype=torch.float32)

        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            target = batch["target"].to(device, dtype=torch.float32).view(-1, 1)
        else:
            target = batch["target"].to(device, dtype=torch.long).view(-1)

        optimizer.zero_grad()
        logits = model(image)
        loss = loss_fn(logits, target)
        val_loss += loss.item()

        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            all_logits.append(logits.view(-1).detach().cpu())
            all_targets.append(target.view(-1).detach().cpu())

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.view(-1) == target.view(-1).long()).sum().item()
            total += target.numel()
        else:
            all_logits.append(logits.detach().cpu())
            all_targets.append(target.detach().cpu())

            preds = torch.argmax(logits, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

        loss.backward()
        gradients = retrieve_gradients(model, gradients, layer)
        batch_count += 1

    val_loss /= len(dataloader)
    acc = correct / total

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        probs = torch.sigmoid(all_logits).detach().cpu().numpy()
        auc = roc_auc_score(all_targets.cpu().numpy(), probs)
    else:
        probs = torch.softmax(all_logits, dim=1).detach().cpu().numpy()
        auc = roc_auc_score(
            all_targets.cpu().numpy(),
            probs,
            multi_class="ovr",
            average="macro"
        )
    if (hook == "premerge" and cid == "gm") or (hook == "postmerge" and cid != "gm"):
        all_logits = None

    gradients = average_gradients(gradients, layer, batch_count)

    optimizer.zero_grad(set_to_none=True)
    log.log_validate_class(cid, acc, val_loss, round, hook)
    struct.registered_hooks[cid] = clear_hooks(struct.registered_hooks[cid])
    return val_loss, acc, auc, gradients, all_logits


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, log=None, round=None):
    test_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_targets = []
    model.eval()

    for batch in tqdm(dataloader, desc="Testing", leave=False):
        image = batch["image"].to(device, dtype=torch.float32)

        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            target = batch["target"].to(device, dtype=torch.float32).view(-1, 1)
        else:
            target = batch["target"].to(device, dtype=torch.long).view(-1)

        logits = model(image)
        loss = loss_fn(logits, target)
        test_loss += loss.item()

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

    log.log_test_class(acc, roc_auc, pr_auc, best_pr, best_re, best_f1, test_loss, round)
    return test_loss, acc, roc_auc, pr, re
