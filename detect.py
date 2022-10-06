#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings

from pytorch_lightning import Trainer
from data.STLPRNDataModule import STLPRNDataModule
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
from model.STLPRNet import STLPRNet, decode

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import torch
import time
import cv2

warnings.filterwarnings("ignore")


def detect(STLPRN, image):
    image = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    image = (np.transpose(np.float32(image), (2, 0, 1)) - 127.5) * 0.0078125
    data = torch.from_numpy(image).float().unsqueeze(0)
    logits = STLPRN(data)
    preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
    predict, _ = decode(preds, CHARS)  # list of predict output
    return predict[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPR Demo')
    parser.add_argument("-image_dir", help='image dir path', default='data/test/', type=str)
    parser.add_argument("-image_path", help='image dir path', default='data/valid/01거0608-6.jpg', type=str)
    parser.add_argument('--weight', type=str, default='saving_ckpt/last.ckpt', help='path to model weight')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Model
    load_model_start = time.time()
    STLPRN = STLPRNet(
        **vars(args)
    ).load_from_checkpoint(args.weight)
    print("Successful to build network in {:2.3f} seconds".format(time.time() - load_model_start))

    since = time.time()
    img = cv2.imread(args.image_path)
    print("RESULT:{},".format(detect(STLPRN, img)), "model inference in {:2.3f} seconds".format(time.time() - since))

