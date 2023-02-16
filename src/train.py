import sys
import torch
import time
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import metrics
from collections import Counter


def train(dataloader, model, loss_fn, optimizer, device):
    start_time = time.time()
    size = len(dataloader.dataset)
    model.train()
    losses = [] 
    for batch, (left, right, left_emb, right_emb, train_label, label, in_train) in enumerate(dataloader):
        left = left.to(torch.int32).to(device)
        right = right.to(torch.int32).to(device)
        left_emb = left_emb.to(torch.float32).to(device) 
        right_emb = right_emb.to(torch.float32).to(device)
        train_label = train_label.to(torch.int32).to(device)
        in_train = in_train.to(torch.float32).to(device)
        if in_train.sum() > 0:

            logp = model(left, right, left_emb, right_emb) # batchsize,
            #loss = loss_fn(logits, torch.arange(logits.shape[0]).to(device))
            loss = loss_fn(logp, train_label, in_train)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if batch % 100 == 99:
                loss, current = np.mean(losses), batch * dataloader.batch_size
                losses = []
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                sys.stdout.flush()
                model.train()
    end_time = time.time()
    print(f"training time for this epoch = {(end_time - start_time) / 60} mins")
    sys.stdout.flush()

def test(dataloader, model, device):
    start_time = time.time()
    num_batches = len(dataloader)
    model.eval()

    valid_scores = []
    valid_labels = []
    test_scores = []
    test_labels = []
    test_scores_train_neg = []
    test_labels_train_neg = []
    test_scores_test = []
    test_labels_test = []
    with torch.no_grad():
        for batch, (left, right, left_emb, right_emb, train_label, label, in_train) in enumerate(dataloader):
            left = left.to(torch.int32).to(device)
            right = right.to(torch.int32).to(device)
            left_emb = left_emb.to(torch.float32).to(device) 
            right_emb = right_emb.to(torch.float32).to(device)
            logp = model(left, right, left_emb, right_emb).cpu().tolist()
            train_label = train_label.tolist()
            label = label.tolist()
            in_train = in_train.tolist()

            for i in range(len(in_train)):
                if in_train[i]:
                    valid_scores.append(logp[i])
                    valid_labels.append(train_label[i])
                if in_train[i] == 0 or (in_train[i] == 1 and train_label[i] == 0):
                    test_scores.append(logp[i])
                    test_labels.append(label[i])
                if in_train[i] == 1 and train_label[i] == 0:
                    test_scores_train_neg.append(logp[i])
                    test_labels_train_neg.append(label[i])
                if in_train[i] == 0:
                    test_scores_test.append(logp[i])
                    test_labels_test.append(label[i])

    valid_ap = average_precision_score(valid_labels, valid_scores)
    test_ap = average_precision_score(test_labels, test_scores)
    test_ap_train_neg = average_precision_score(test_labels_train_neg, test_scores_train_neg)
    test_ap_test = average_precision_score(test_labels_test, test_scores_test)

    end_time = time.time()
    print(f"testing time = {(end_time - start_time) / 60} mins")
    sys.stdout.flush()

    return {"AP":valid_ap}, {"AP": test_ap, "AP_train_neg": test_ap_train_neg, "AP_test": test_ap_test}

