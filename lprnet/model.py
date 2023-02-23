"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import os
import yaml
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from nltk.metrics.distance import edit_distance
from typing import Tuple
from argparse import Namespace

from lprnet.utils import CTCLabelConverter, AttnLabelConverter
from lprnet.modules.transformation import TPS_SpatialTransformerNetwork
from lprnet.modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor,
)
from lprnet.modules.sequence_modeling import BidirectionalLSTM
from lprnet.modules.prediction import Attention


class Model(pl.LightningModule):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction,
        }

        """ Criterion """
        if "CTC" in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3

        """ Transformation """
        if opt.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel,
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if opt.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(
                    self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size
                ),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            )
            self.SequenceModeling_output = opt.hidden_size
        else:
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == "Attn":
            self.Prediction = Attention(
                self.SequenceModeling_output, opt.hidden_size, opt.num_class
            )
        else:
            raise Exception("Prediction is neither CTC or Attn")

        # weight initialization
        for name, param in self.named_parameters():
            if "localization_fc2" in name:
                print(f"Skip {name} as it is already initialized")
                continue
            try:
                if "bias" in name:
                    init.constant_(param, 0.0)
                elif "weight" in name:
                    init.kaiming_normal_(param)
            except Exception:  # for batchnorm.
                if "weight" in name:
                    param.data.fill_(1)
                continue

        """ setup loss """
        if "CTC" in opt.Prediction:
            self.criterion = torch.nn.CTCLoss(zero_infinity=True)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.save_hyperparameters(opt)

    def forward(self, input, text=None, is_train=True):
        """Transformation stage"""
        if not self.stages["Trans"] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2)
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            if text == None:
                batch_size = input.size(0)
                text = torch.LongTensor(
                    batch_size, self.opt.batch_max_length + 1
                ).fill_(0)

            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=self.opt.batch_max_length,
            )

        return prediction

    def configure_optimizers(self):
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, self.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print("Trainable params num : ", sum(params_num))

        # setup optimizer
        if self.opt.adam:
            return optim.Adam(
                filtered_parameters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
        else:
            return optim.Adadelta(
                filtered_parameters, lr=self.opt.lr, rho=self.opt.rho, eps=self.opt.eps
            )

    def training_step(self, batch, batch_idx):
        # train part
        image_tensors, labels = batch
        image = image_tensors
        text, length = self.converter.encode(
            labels, batch_max_length=self.opt.batch_max_length
        )
        batch_size = image.size(0)

        if "CTC" in self.opt.Prediction:
            preds = self(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.log_softmax(2).permute(1, 0, 2)
            cost = self.criterion(preds, text, preds_size, length)
        else:
            preds = self(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = self.criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
            )

        self.log("train-loss", cost, prog_bar=True, batch_size=self.opt.batch_size)

        return cost

    def validation_step(self, batch, batch_idx):
        n_correct = 0
        norm_ED = 0
        image_tensors, labels = batch
        batch_size = image_tensors.size(0)
        image = image_tensors

        # For max length prediction
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size)
        text_for_pred = torch.LongTensor(
            batch_size, self.opt.batch_max_length + 1
        ).fill_(0)

        text_for_loss, length_for_loss = self.converter.encode(
            labels, batch_max_length=self.opt.batch_max_length
        )

        if "CTC" in self.opt.Prediction:
            preds = self(image, text_for_pred)

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = self.criterion(
                preds.log_softmax(2).permute(1, 0, 2),
                text_for_loss,
                preds_size,
                length_for_loss,
            )

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self(image, text_for_pred, is_train=False)

            preds = preds[:, : text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = self.criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1),
            )

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)

        self.log_dict(
            {"val-loss": cost},
            prog_bar=True,
            batch_size=self.opt.batch_size,
        )

        return preds, preds_str, labels

    def validation_epoch_end(self, outputs):
        # preds = torch.stack(outputs)
        preds = []
        preds_str = []
        labels = []
        n_correct = 0
        norm_ED = 0

        for pred, pred_str, label in outputs:
            preds.extend(pred)
            preds_str.extend(pred_str)
            labels.extend(label)

        preds = torch.stack(preds)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        print("=" * 15 * 3)
        print(f"{'predict':15s}{'ground truth':15s}{'confidence':15s}")
        print("=" * 15 * 3)
        for i, (gt, pred, pred_max_prob) in enumerate(
            zip(labels, preds_str, preds_max_prob)
        ):
            if "Attn" in self.opt.Prediction:
                gt = gt[: gt.find("[s]")]
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if self.opt.sensitive and self.opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = "0123456789abcdefghijklmnopqrstuvwxyz"
                out_of_alphanumeric_case_insensitve = (
                    f"[^{alphanumeric_case_insensitve}]"
                )
                pred = re.sub(out_of_alphanumeric_case_insensitve, "", pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, "", gt)

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

            if i < 10:
                print(f"{pred:15s}{gt:15s}{confidence_score:.4f}")

        accuracy = n_correct / float(len(preds)) * 100
        norm_ED = norm_ED / float(len(preds))  # ICDAR2019 Normalized Edit Distance

        print("-" * 15 * 3)
        print(
            f"Accuracy: {accuracy:.4f}, Norm_ED: {norm_ED:.4f}, Confidence: {sum(confidence_score_list)/len(confidence_score_list):.4f}"
        )

        self.log_dict(
            {"val-acc": accuracy, "val-ned": norm_ED},
            prog_bar=True,
            batch_size=self.opt.batch_size,
        )

    def test_step(self, batch, batch_idx):
        n_correct = 0
        norm_ED = 0
        image_tensors, labels = batch
        batch_size = image_tensors.size(0)
        image = image_tensors
        # For max length prediction
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size)
        text_for_pred = torch.LongTensor(
            batch_size, self.opt.batch_max_length + 1
        ).fill_(0)

        text_for_loss, length_for_loss = self.converter.encode(
            labels, batch_max_length=self.opt.batch_max_length
        )

        if "CTC" in self.opt.Prediction:
            preds = self(image, text_for_pred)

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = self.criterion(
                preds.log_softmax(2).permute(1, 0, 2),
                text_for_loss,
                preds_size,
                length_for_loss,
            )

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self(image, text_for_pred, is_train=False)

            preds = preds[:, : text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = self.criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1),
            )

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if "Attn" in self.opt.Prediction:
                gt = gt[: gt.find("[s]")]
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if self.opt.sensitive and self.opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = "0123456789abcdefghijklmnopqrstuvwxyz"
                out_of_alphanumeric_case_insensitve = (
                    f"[^{alphanumeric_case_insensitve}]"
                )
                pred = re.sub(out_of_alphanumeric_case_insensitve, "", pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, "", gt)

            if pred == gt:
                n_correct += 1

            """
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            """

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

            # if batch_idx == 0:
            #     print(f'{pred}, {gt}, {pred==gt}, {confidence_score.item():.5f}')

        accuracy = n_correct / float(batch_size) * 100
        norm_ED = norm_ED / float(batch_size)  # ICDAR2019 Normalized Edit Distance

        self.log_dict(
            {"test-loss": cost, "test-accuracy": accuracy, "test-norm_ED": norm_ED},
            prog_bar=True,
            batch_size=self.opt.batch_size,
        )

    def predict_step(self, batch, batch_idx):
        image_tensors, image_path_list = batch
        batch_size = image_tensors.size(0)
        image = image_tensors
        # For max length prediction
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size)
        text_for_pred = torch.LongTensor(
            batch_size, self.opt.batch_max_length + 1
        ).fill_(0)

        if "CTC" in self.opt.Prediction:
            preds = self(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(preds_index, preds_size)

        else:
            preds = self(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        predicts = []
        for img_name, pred, pred_max_prob in zip(
            image_path_list, preds_str, preds_max_prob
        ):
            if "Attn" in self.opt.Prediction:
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # img_name to id
            img_name = img_name.split("/")[-1].split(".png")[0]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = (
                pred_max_prob.cumprod(dim=0)[-1]
                if len(pred_max_prob.cumprod(dim=0)) > 0
                else 0.0
            )

            predicts.append((img_name, pred, confidence_score.item()))

        return predicts

    def imread(self, image, device):
        """
        Read Texts in PIL Image.

        Args:
            image: PIL Image to Read
            device: torch.device

        Returns:
            predict: predicted string
            confidence: model's confidence_score
            inference_time: inference_time in ms
        """
        from PIL import Image
        import torchvision.transforms as transforms
        import time

        start_time = time.time()

        totensor = transforms.ToTensor()

        image = image.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)
        image = totensor(image).to(device)
        image.sub_(0.5).div_(0.5)
        image = image.unsqueeze(0)

        # For max length prediction
        length_for_pred = torch.IntTensor([self.opt.batch_max_length])
        text_for_pred = torch.LongTensor(1, self.opt.batch_max_length + 1).fill_(0)

        if "CTC" in self.opt.Prediction:
            preds = self(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)])
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(preds_index, preds_size)

        else:
            preds = self(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if "Attn" in self.opt.Prediction:
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = (
                pred_max_prob.cumprod(dim=0)[-1]
                if len(pred_max_prob.cumprod(dim=0)) > 0
                else 0.0
            )

        end_time = time.time()

        return pred.upper(), confidence_score.item(), (end_time - start_time) * 1000


def load_LPRNet(yaml_path: str) -> Tuple[Model, Namespace]:
    """
    Load LPRNet model from yaml file
    :param yaml_path: path to yaml file
    :return: loaded LPRNet model, Namespace object with model parameters
    """

    # load configuration
    with open(yaml_path, "r") as f:
        opt = yaml.safe_load(f)
        opt = Namespace(**opt)
        print(f"Configuration loaded from {yaml_path}")

    # set experiment name if not exists
    if not opt.exp_name:
        opt.exp_name = f"{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}"
        opt.exp_name += f"-Seed{opt.manualSeed}"

    # Seed and GPU setting
    pl.seed_everything(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # Load model from checkpoint if saved_model is exists else create new model
    if opt.saved_model:
        if not os.path.exists(opt.saved_model):
            raise FileNotFoundError(f"{opt.saved_model} is not exist!")

        model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
        print(f"Model loaded from {opt.saved_model}")
    else:
        model = Model(opt)
        print(f"Model creadted with {yaml_path}")

    # model warm-up with dummy tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    warm_up_image = torch.rand(1, 3, opt.imgH, opt.imgW).to(device)
    warm_up_text = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

    model(warm_up_image, warm_up_text)

    return model, opt
