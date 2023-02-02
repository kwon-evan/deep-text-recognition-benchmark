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
from rich import print
from PIL import Image

from LPRNet import Model

warnings.filterwarnings(action='ignore')

def predict(opt):
    if opt.saved_model == '' or os.path.exists(opt.saved_model):
        assert f'{opt.saved_model} is not exist!'

    model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
    model.eval().to(device)
    print(f'model loaded from checkpoint {opt.saved_model}')

    print(model.hparams)


    for img_name in os.listdir('demo_images'):
        img = Image.open(f'demo_images/{img_name}')
        label = img_name.split('.')[0].split('-')[0]

        pred, conf, time = model.imread(img, device)
        print(f'''
        ------------------------
        label: {label}
        pred : {pred}
        correct : {label==pred}
        conf : {conf:.5f}
        time : {time:.5f} ms
        ------------------------
        ''')


if __name__ == '__main__':
    """ load configuration """
    with open('config.yaml', 'r') as f:
        opt = yaml.safe_load(f)
        opt = Namespace(**opt)

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

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
