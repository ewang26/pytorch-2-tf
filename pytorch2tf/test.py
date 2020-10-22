import argparse
from .arch_convert import model_arch_conversion
from .weight_convert import pytorch_weights_to_tf_weights

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, help='path to .pth file/.pth.tar file')
parser.add_argument('--out_path', type=str, help='path to output saved model architecture')
parser.add_argument('--arch', type=str, help='architecture type of pretrained model')
#TODO support custom architecture definitions

args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path
arch = args.arch
