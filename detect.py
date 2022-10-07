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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPR Demo')
    parser.add_argument("-image_dir", help='image dir path', default='data/test/', type=str)
    parser.add_argument("-image_path", help='image dir path', default='data/valid/01거0608-6.jpg', type=str)
    parser.add_argument('--weight', type=str, default='saving_ckpt1/best.ckpt', help='path to model weight')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Model
    load_model_start = time.time()
    STLPRN = STLPRNet(
        **vars(args)
    ).load_from_checkpoint(args.weight)
    print("Successful to build network in {:2.3f}ms".format(1E3*(time.time() - load_model_start)))

    img = cv2.imread(args.image_path)
    img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_CUBIC)

    since = time.time()
    pred = STLPRN.detect(img)
    end = time.time()

    print("RESULT:{},".format(pred), "model inference in {:2.3f}ms".format(1E3*(end - since)))

