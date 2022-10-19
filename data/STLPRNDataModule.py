import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from STLPRNet.data.load_data import ImgDataLoader, LPRDataLoader, collate_fn


class STLPRNDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data_dir: str = "./data/train",
                 val_data_dir: str = "./data/valid",
                 test_data_dir: str = "./data/test",
                 predict_images=None,
                 img_size=(94, 24),
                 batch_size: int = 512):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.predict_images = predict_images
        self.img_size = img_size
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit" or stage == "train":
            self.train = LPRDataLoader([self.train_data_dir], self.img_size)
            self.val = LPRDataLoader([self.val_data_dir], self.img_size)

        if stage == "val":
            self.val = LPRDataLoader([self.val_data_dir], self.img_size)

        if stage == "test":
            self.test = LPRDataLoader([self.test_data_dir], self.img_size)

        if stage == "predict":
            self.predict = ImgDataLoader(self.predict_images, self.img_size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
