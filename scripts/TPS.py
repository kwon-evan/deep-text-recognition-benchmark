import os
import warnings

import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms

from lprnet import load_LPRNet

warnings.filterwarnings(action="ignore")


def TPS(model, opt):
    # IMAGE_FOLDER = '/home/fourind/projects/datas/kor-plates/test'
    IMAGE_FOLDER = "demo_images"
    for img_name in sorted(os.listdir(IMAGE_FOLDER)):
        image = Image.open(f"{IMAGE_FOLDER}/{img_name}")
        image_size = image.size
        label = img_name.split(".")[0].split("-")[0]

        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        image = image.resize((opt.imgW, opt.imgH), Image.ANTIALIAS)
        image = to_tensor(image).to(device).unsqueeze(0)
        # image.sub_(0.5).div_(0.5) # Image Normalize

        """ Transformation stage """
        if not model.stages["Trans"] == "None":
            warped = model.Transformation(image)

        warped = warped.squeeze(0)
        warped = to_pil(warped)
        warped = warped.resize(image_size, Image.ANTIALIAS)
        rename = img_name.split(".")[0] + "-warped.jpg"
        warped.save(f"{IMAGE_FOLDER}/{rename}", dpi=(200, 200))


if __name__ == "__main__":
    """load configuration"""
    model, opt = load_LPRNet("config-idn.yaml")
    model.eval().freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TPS(model, opt)
