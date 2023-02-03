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
import pandas as pd
from rich import print

from lprnet import Model
from lprnet import LMDBDataModule

warnings.filterwarnings(action='ignore')

def predict(opt):
    if opt.saved_model == '' or os.path.exists(opt.saved_model):
        assert f'{opt.saved_model} is not exist!'

    dm = LMDBDataModule(opt)
    model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
    model.eval()
    print(f'model loaded from checkpoint {opt.saved_model}')

    print(model.hparams)

    trainer = pl.Trainer(
        accelerator='auto',
        devices=opt.num_gpu,
        precision=16,
    )

    predict_result = trainer.predict(model, dm)

    predict_result = [pred for pred in predict_result[0]]
    predict_df = pd.DataFrame([(img_name,
                                    img_name.split('.jpg')[0].split('-')[0],
                                    pred.upper(),
                                    conf) for img_name, pred, conf in predict_result],
                                  columns=['img_name', 'label', 'pred', 'conf'])
    predict_df['correct'] = predict_df.apply(lambda x: x.label == x.pred, axis=1)
    predict_df.to_csv('predict_result.csv', index=False)
    failures = predict_df.loc[predict_df['correct'] == False]
    failures.to_csv('predict_failures.csv', index=False)
    print('Accuracy:', (len(predict_df) - len(failures)) / len(predict_df) * 100)

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

    predict(opt)
