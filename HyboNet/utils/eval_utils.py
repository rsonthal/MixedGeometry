from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import torch

def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1

def prf(output, labels):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    return precision, recall, f1

def prf_lp(output, labels):
    probs = torch.tensor(output).sigmoid()
    preds = probs.round().tolist()
    # if preds.is_cuda:
    #     preds = preds.cpu()
    #     labels = labels.cpu()
    precision = precision_score(labels, preds, average='macro', zero_division=0.0)
    recall = recall_score(labels, preds, average='macro', zero_division=0.0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0.0)
    return precision, recall, f1

class MarginLoss(torch.nn.Module):
    def __init__(self, margin) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, preds):
        correct_preds = preds[..., 0:1]
        return torch.nn.functional.relu(self.margin - correct_preds + preds).mean()