from typing import List
import numpy as np


def encode(imgname: str, chars: List[str]):
    chars_dict = {char: i for i, char in enumerate(chars)}
    label = []

    i = 0
    while i < len(imgname):
        j = len(imgname)
        while i < j and not imgname[i:j] in chars:
            j -= 1

        if imgname[i:j] in chars:
            label.append(chars_dict[imgname[i:j]])
            i=j
        else:
            assert 0, f'no such char in {imgname}'

    return label

def decode(preds, chars):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = ''
        # pre_c = pred_label[0]
        for c in pred_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(chars) - 1):
                if c == len(chars) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)

    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += chars[i]
        labels.append(lb)

    return labels, pred_labels


def accuracy(logits, labels, lengths, chars):
    preds = logits.cpu().detach().numpy()
    _, pred_labels = decode(preds, chars)

    TP, total = 0, 0
    start = 0
    for i, length in enumerate(lengths):
        label = labels[start:start + length]
        start += length
        if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
            TP += 1
        total += 1

    return TP / total