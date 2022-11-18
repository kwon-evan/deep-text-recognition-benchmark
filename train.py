import os
import sys
import argparse
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
import warnings

sys.path.append('.')
from model.model import Model
from data.data_module import DataModule

warnings.filterwarnings("ignore")
wandb_logger = WandbLogger(project="LPRNet")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STLPRN Training')
    parser.add_argument('--img_dirs_train', default="./data/train", help='the training images path')
    parser.add_argument('--img_dirs_val', default="./data/valid", help='the validation images path')
    parser.add_argument('--save_dir', type=str, default='saving_ckpt', help='directory to save check points')
    parser.add_argument('--img_size', default=(100, 50), help='the image size')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=2e-5, help='STN adam optimizer weight_decay')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    args.save_dir += now.strftime("_%m-%d_%H:%M")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    STLPRNet = Model(**vars(args))
    print("STLPRNet loaded")

    # Set Data Module
    data_module = DataModule(train_data_dir=args.img_dirs_train,
                                   val_data_dir=args.img_dirs_val,
                                   img_size=args.img_size,
                                   batch_size=args.batch_size)

    # Callbacks
    chk_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='epoch:{epoch:02d}-{val-acc:.3f}',
        verbose=True,
        save_last=True,
        save_top_k=5,
        monitor='val-loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(monitor='val-loss', min_delta=0.00, patience=50, verbose=True, mode='min')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # Set Trainer
    trainer = Trainer.from_argparse_args(
        args,
        # auto_lr_find=True,
        # auto_scale_batch_size="binsearch",
        callbacks=[chk_callback, early_stop_callback, lr_monitor_callback, RichProgressBar()],
        precision=16,
        accelerator="gpu",
        amp_backend="apex",
        devices=1,
        logger=wandb_logger
    )
    # trainer.tune(STLPRNet)

    # Train
    print('training kicked off..')
    print('-' * 10)
    trainer.fit(model=STLPRNet, datamodule=data_module)
