import numpy as np
import torch
import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def get_device_torch():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def model_metrics_evaluate(model, val_dataloader, matrix_route=""):
    device = get_device_torch()

    all_probs = []
    labels = []
    y_pred = []
    y_label = []

    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        probs = F.softmax(logits, dim=1).cpu().numpy()
        for i, prob in enumerate(probs):
            labels.append(b_labels[i])
            all_probs.append(prob)

        preds = torch.argmax(logits, dim=1).cpu().flatten()

        for pred in enumerate(preds):
            y_pred.append(int(pred[1].numpy()))

    for i, probs in enumerate(all_probs):
        y_label.append(int(labels[i].cpu().numpy()))

    f1 = f1_score(y_label, y_pred, average="macro")
    accuracy = accuracy_score(y_label, y_pred)
    precision = precision_score(y_label, y_pred, average="macro")
    recall = recall_score(y_label, y_pred, average="macro")

    report = classification_report(y_label, y_pred)
    print(report)

    return accuracy, precision, recall, f1