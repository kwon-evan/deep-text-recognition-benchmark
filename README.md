# What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis

| [paper](https://arxiv.org/abs/1904.01906) | [training and evaluation data](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | [failure cases and cleansed label](https://github.com/clovaai/deep-text-recognition-benchmark#download-failure-cases-and-cleansed-label-from-here) | [pretrained model](https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=0) | [Baidu ver(passwd:rryk)](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) |

## Rewritten with pytorch-lighting

⚠️ This is a rewritten repository for personal use for the purpose of licence plate recognition. This will not be no longer maintained.<br><br>
[origianl repository](https://github.com/clovaai/deep-text-recognition-benchmark)<br><br>

## Getting Started

### Dependency

r requirements : pytorch-lightning, lmdb, pillow, torchvision, nltk, natsort

```shell
python3 setup.py install
```

### Training and evaluation

Try to train and test our best accuracy model TRBA (**T**PS-**R**esNet-**B**iLSTM-**A**ttn) also. ([download pretrained model](https://drive.google.com/file/d/1-oVujDx3bREgDx5lQ9C0VXXbI330YAth/view?usp=share_link))
   - you can modify the `config.yaml` for your own training configuration

```shell
python scripts/train.py
```

```shell
python scripts/test.py
```

```shell
python scripts/predict.py
```

- data structure
```
data/
 ├ AA0000BB-0.jpg
 ├ AA0000BB-1.jpg
 ├ AA0000BB-2.jpg
  ...
```

## Acknowledgements

This implementation has been based on these repository [crnn.pytorch](https://github.com/meijieru/crnn.pytorch), [ocr_attention](https://github.com/marvis/ocr_attention), [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

## Citation

Please consider citing this work in your publications if it helps your research.

```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  pubstate={published},
  tppubtype={inproceedings}
}
```

## License

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
