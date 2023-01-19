from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import os
import warnings
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time
from cv2 import cv2

from model.model import Model

warnings.filterwarnings("ignore")

def transform(img):
    img = cv2.resize(img, dsize=(100, 50), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32)
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor([img])

def inv_normalize(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1, 2, 0))
    inp = 127.5 + inp / 0.0078125
    inp = inp.astype('uint8')

    return inp

def resize_pad(img):
    img_size = (100, 50)

    base_pic=np.zeros((img_size[1],img_size[0],3),np.uint8)
    h,w=img.shape[:2]
    ash=img_size[1]/h
    asw=img_size[0]/w
    if asw<ash:
        sizeas=(int(w*asw),int(h*asw))
    else:
        sizeas=(int(w*ash),int(h*ash))
    img = cv2.resize(img,dsize=sizeas)
    base_pic[int(img_size[1]/2-sizeas[1]/2):int(img_size[1]/2+sizeas[1]/2),
             int(img_size[0]/2-sizeas[0]/2):int(img_size[0]/2+sizeas[0]/2),:]=img

    return base_pic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPR Demo')
    parser.add_argument("-image_dir", help='image dir path', default='data/normal/비사업용-세자리/', type=str)
    parser.add_argument('--weight', type=str, default='saving_ckpt_11-18_16:59/epoch=30-val-acc=0.953.ckpt', help='path to model weight')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    load_model_start = time.time()
    model = Model().load_from_checkpoint(args.weight).to(device)
    print(f"Successful to build network in {time.time() - load_model_start}s")

    images = [cv2.imread(args.image_dir + image_name) 
              for image_name in os.listdir(args.image_dir)[:40]]

    labels = [model.detect(img, device=device) for img in images]

    for i, label in enumerate(labels):
        suffix = '\n' if i % 5 == 4 else '\t'
        print(f'{label:>10s}', end=suffix)
    else:
        print()

    target_layer = model.LPRNet.container
    cam = EigenCAM(model, target_layer, use_cuda=True)

    lines = []
    for image in tqdm(images):
        image = resize_pad(image)
        tensor = transform(image)
        grayscale_cam = cam(tensor)[0, :, :]

        affine_image = model.STN(tensor.to(device))
        affine_image = inv_normalize(affine_image)

        affine_norm = (affine_image - affine_image.min()) / (affine_image.max() - affine_image.min()).astype(np.float32)
        result = show_cam_on_image(affine_norm, grayscale_cam, use_rgb=True)

        lines.append(cv2.hconcat([affine_image, result]))

    output = cv2.vconcat([cv2.hconcat(lines[i:i+5]) for i in range(0, len(lines), 5)])
    cv2.imwrite('result2.jpg', output)
    cv2.imshow('result', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


