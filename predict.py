#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from argparse import Namespace
import warnings
import yaml
import torch
from cv2 import cv2
from rich import print
from rich.progress import track
from sklearn.metrics import accuracy_score

from lprnet.lprnet import LPRNet

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    with open('config/kor_config.yaml') as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    load_model_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprnet = LPRNet(args)
    lprnet.load_state_dict(torch.load(args.pretrained))
    lprnet.to(device).eval()
    print(f"Successful to build network in {time.time() - load_model_start}s")

    imgs = os.listdir(args.test_dir)[:100]
    labels = [n.split('.')[0].split('-')[0] for n in track(imgs)]
    preds = [lprnet.detect(cv2.imread(args.test_dir + img), device) for img in track(imgs)]

    print("Accuracy: ", accuracy_score(labels, preds))
