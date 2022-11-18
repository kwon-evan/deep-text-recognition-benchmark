#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import warnings
from pytorch_lightning import Trainer
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import torch
import time
import cv2
sys.path.append('.')
from model.model import Model
from data.load_data import ImgDataLoader
from data.data_module import DataModule

warnings.filterwarnings("ignore")


def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1, 2, 0))
    inp = 127.5 + inp / 0.0078125
    inp = inp.astype('uint8')

    return inp


def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansKR-Medium.otf", textSize, encoding="encoding=unic")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        for c in pred_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)

    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)

    return labels, pred_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPR Demo')
    parser.add_argument("-image_dir", help='image dir path', default='data/test/', type=str)
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--weight', type=str, default='saving_ckpt/best.ckpt', help='path to model weight')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image1 = cv2.imread('data/valid/300두4673.jpg')
    image2 = cv2.imread('data/valid/99루4030.jpg')
    cv2.imshow('1', image1)
    cv2.imshow('2', image2)
    images = [image1, image2]

    load_model_start = time.time()
    STLPRN = STLPRNet(
        **vars(args)
    ).load_from_checkpoint(args.weight)
    print(f"Successful to build network in {time.time() - load_model_start}s")

    predicts = []
    since = time.time()
    for img, label, leng in ImgDataLoader(images):
        image = (np.transpose(np.float32(img), (0, 1, 2)) - 127.5) * 0.0078125
        data = torch.from_numpy(image).float().unsqueeze(0)
        logits = STLPRN(data)
        preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
        predict, _ = decode(preds, CHARS)  # list of predict output
        predicts.append(predict[0])
    time_total = time.time() - since
    print(predicts)
    print("model inference in {:2.3f} seconds".format(time_total))
    print(f"img/s: {time_total / 2}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
