import numpy as np
import torch
from torchvision.ops import masks_to_boxes
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve


def precision_recall_f1(metrics, prf1):
    for t, vals in metrics.items():
        tp = vals["TP"]
        fp = vals["FP"]
        fn = vals["FN"]

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall /
             (precision + recall)) if (precision + recall) > 0 else 0.0

        prf1[t]["precision"] = precision
        prf1[t]["recall"] = recall
        prf1[t]["f1"] = f1
    return prf1


def batch_prf1_metrics(gt, pred, metrics):
    for t in metrics.keys():
        for x in range(gt.shape[0]):
            gt_box = torch.empty((0, 4), device=gt.device) if gt[x].sum() == 0 else masks_to_boxes(gt[x].squeeze(1))

            pred_mask = (pred[x] >= t)
            pred_box = torch.empty((0, 4), device=pred_mask.device) if pred_mask.sum() == 0 else masks_to_boxes(pred_mask.squeeze(1))

            # No GT lesion
            if gt_box.shape[0] == 0:
                metrics[t]["FP"] += int(pred_box.shape[0] > 0)
                continue

            # GT exists, no prediction
            if pred_box.shape[0] == 0:
                metrics[t]["FN"] += 1
                continue

            # One GT box, one pred box
            iou = box_iou(gt_box[0], pred_box[0])

            if iou >= 0.5:
                metrics[t]["TP"] += 1
            else:
                metrics[t]["FP"] += 1
                metrics[t]["FN"] += 1
    return metrics
