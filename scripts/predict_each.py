import os
import warnings

import torch
import torch.utils.data
import pandas as pd
from rich import print
from PIL import Image

from lprnet import load_LPRNet

warnings.filterwarnings(action="ignore")


def predict(model, opt):
    print(model.hparams)

    # IMAGE_FOLDER = '/home/fourind/projects/datas/kor-plates/test'
    IMAGE_FOLDER = "demo_images/"
    result = {
        "img_names": [],
        "labels": [],
        "preds": [],
        "confs": [],
        "times": [],
        "acc": [],
    }
    for img_name in os.listdir(IMAGE_FOLDER):
        img = Image.open(f"{IMAGE_FOLDER}/{img_name}")
        label = img_name.split(".")[0].split("-")[0]

        pred, conf, time = model.imread(img)
        result["img_names"].append(img_name)
        result["labels"].append(label)
        result["preds"].append(pred)
        result["confs"].append(conf)
        result["times"].append(time)
        result["acc"].append(label == pred)

    df = pd.DataFrame(result)
    print(df)


if __name__ == "__main__":
    """load configuration"""
    model, opt = load_LPRNet("config-idn.yaml")
    model.eval().freeze()

    predict(model, opt)
