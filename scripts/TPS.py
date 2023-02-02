import os
from argparse import Namespace
import yaml
import warnings

import torch
import torch.utils.data
from rich import print
from PIL import Image
import torchvision.transforms as transforms

from lprnet import Model

warnings.filterwarnings(action='ignore')

def TPS(opt):
    if opt.saved_model == '' or os.path.exists(opt.saved_model):
        assert f'{opt.saved_model} is not exist!'

    model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
    model.eval().to(device)
    print(f'model loaded from checkpoint {opt.saved_model}')

    for img_name in sorted(os.listdir('demo_images')):
        image = Image.open(f'demo_images/{img_name}')
        label = img_name.split('.')[0].split('-')[0]

        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        image = image.resize((opt.imgW, opt.imgH), Image.ANTIALIAS)
        image = to_tensor(image).to(device).unsqueeze(0)
        # image.sub_(0.5).div_(0.5) # Image Normalize

        """ Transformation stage """
        if not model.stages['Trans'] == "None":
            warped = model.Transformation(image)

        warped = warped.squeeze(0)
        warped = to_pil(warped)
        warped = warped.resize((512, 200), Image.ANTIALIAS)
        rename = img_name.split('.')[0] + '-warped.jpg'
        warped.save(f'demo_images/{rename}', dpi=(200, 200))


if __name__ == '__main__':
    """ load configuration """
    with open('config.yaml', 'r') as f:
        opt = yaml.safe_load(f)
        opt = Namespace(**opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TPS(opt)
