import yaml
from argparse import Namespace

with open('args.yaml') as f:
    args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

