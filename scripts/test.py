import os
import random
import string
from argparse import Namespace
import yaml
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import pytorch_lightning as pl
import numpy as np
from rich import print

from lprnet import Model
from lprnet import LMDBDataModule

warnings.filterwarnings(action='ignore')

def test(opt):
    if opt.saved_model == '' or os.path.exists(opt.saved_model):
        assert f'{opt.saved_model} is not exist!'

    dm = LMDBDataModule(opt)
    print("Loding Saved:", opt.saved_model)
    print(os.path.exists(opt.saved_model))
    model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
    model.eval()
    print(f'model loaded from checkpoint {opt.saved_model}')

    print(model.hparams)

    trainer = pl.Trainer(
        accelerator='auto',
        devices=opt.num_gpu,
        precision=16,
    )

    test_result = trainer.test(model, dm)

    print(test_result)


if __name__ == '__main__':
    """ load configuration """
    with open('config-idn.yaml', 'r') as f:
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

    test(opt)
