import os
import warnings

import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms

from trba import load_LPRNet

warnings.filterwarnings(action="ignore")


def TPS(model, opt):
    # SRC = '/home/fourind/projects/datas/kor-plates/test'
    SRC = "demo_images"
    DST = "warped"
    os.makedirs(DST, exist_ok=True)
    for img_name in sorted(os.listdir(SRC)):
        image = Image.open(f"{SRC}/{img_name}")
        image_size = image.size

        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        image = image.resize((opt.imgW, opt.imgH), Image.ANTIALIAS)
        image = to_tensor(image).to(device).unsqueeze(0)
        image.sub_(0.5).div_(0.5)  # Image Normalize

        """ Transformation stage """
        warped = model.Transformation(image)

        warped.mul_(0.5).add_(0.5)  # Image Denormalize
        warped = warped.squeeze(0)
        warped = to_pil(warped)
        warped = warped.resize(image_size, Image.ANTIALIAS)
        warped.save(f"{DST}/{img_name}", dpi=(200, 200))


if __name__ == "__main__":
    """load configuration"""
    model, opt = load_LPRNet("config-idn.yaml")
    model.eval().freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TPS(model, opt)
