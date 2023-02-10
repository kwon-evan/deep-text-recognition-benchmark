import os
import random
import string
from argparse import Namespace
import yaml
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import pandas as pd
from rich import print
from PIL import Image

from lprnet import Model

warnings.filterwarnings(action='ignore')

def predict(opt):
    if opt.saved_model == '' or os.path.exists(opt.saved_model):
        assert f'{opt.saved_model} is not exist!'

    model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
    model.eval().to(device)
    # model.freeze()
    print(f'model loaded from checkpoint {opt.saved_model}')

    print(model.hparams)

    # IMAGE_FOLDER = '/home/fourind/projects/datas/kor-plates/test'
    IMAGE_FOLDER = 'demo_images/'
    result = {
        'img_names': [],
        'labels': [],
        'preds': [],
        'confs': [],
        'times': [],
        'acc': []
    }
    for img_name in os.listdir(IMAGE_FOLDER):
        img = Image.open(f'{IMAGE_FOLDER}/{img_name}')
        label = img_name.split('.')[0].split('-')[0]

        pred, conf, time = model.imread(img, device)
        result['img_names'].append(img_name)
        result['labels'].append(label)
        result['preds'].append(pred)
        result['confs'].append(conf)
        result['times'].append(time)
        result['acc'].append(label==pred)

    df = pd.DataFrame(result)
    print(df)


if __name__ == '__main__':
    """ load configuration """
    with open('config-idn.yaml', 'r') as f:
        opt = yaml.safe_load(f)
        opt = Namespace(**opt)

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predict(opt)
