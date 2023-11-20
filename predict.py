import argparse
import glob
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask, overlay_mask_on_image
from unet import UNet
from deeplab import DeepLabv3Plus
from FCN.FCN import get_fcn_model as FCN
from PSPNet.PSPNet import PSPNet
from SegNet.SegNet import SegNet


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5,
                target_size=(None, None)):
    assert target_size[0] is not None and target_size[1] is not None, 'target_size must be specified as (height, width)'

    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, is_mask=False, target_h=target_size[0], target_w=target_size[1]))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


class Args:
    '''Class to hold the arguments passed to the predict.py script'''

    def __init__(self):
        self.model = './checkpoints/model.pth'
        self.image_dir = './mock_dataset/val'  ## directory containing the images to be predicted
        self.viz = False
        self.mask_threshold = 0.5
        self.number_of_in_channels = 3
        self.classes = 3
        self.target_size = (512, 512)
        self.cnn_arch = 'deeplabv3+'
        self.backbone = 'resnet34'
        self.bilinear = True


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = Args()

    in_files = [os.path.normpath(f).replace(os.sep, '/') for f in
                glob.glob(os.path.join(args.image_dir, '**/*.jpg'), recursive=True)]

    print('found {} .jpg files in {}'.format(len(in_files), args.image_dir))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.cnn_arch.lower() == 'unet':
        print('Using UNet ...')
        net = UNet(n_channels=args.number_of_in_channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.cnn_arch.lower() == 'deeplabv3+':
        print('Using DeepLab ...')
        assert args.backbone.lower() in ['resnet18', 'resnet34', 'resnet50',
                                         'resnet101'], 'backbone must be one of resnet18, resnet50, resnet101 when using DeepLab'
        net = DeepLabv3Plus(n_channels=args.number_of_in_channels, n_classes=args.classes, os=16,
                            pretrained=False, _print=True, backbone=args.backbone,
                            bilinear=args.bilinear)
    elif args.cnn_arch.lower() in ['fcn32s', 'fcn16s', 'fcn8s', 'fcns']:
        print('Using FCN ...')
        assert args.backbone in ['vgg11', 'vgg13', 'vgg16',
                                 'vgg19'], 'backbone must be one of vgg11, vgg13, vgg16, vgg19 when using FCN'

        net = FCN(n_channels=args.number_of_in_channels, n_classes=args.classes, backbone=args.backbone,
                  pretrained=False, bilinear=args.bilinear, arch_type=args.cnn_arch,
                  requires_grad=True)
    elif args.cnn_arch.lower() == 'pspnet':
        print('Using PSPNet ...')
        assert args.backbone in ['resnet18', 'resnet34', 'resnet50',
                                 'resnet101'], 'backbone must be one of resnet18, resnet50, resnet101 when using PSPNet'
        net = PSPNet(n_channels=args.number_of_in_channels, n_classes=args.classes, os=16, backbone=args.backbone,
                     pretrained=False, bilinear=args.bilinear)
    elif args.cnn_arch.lower() == 'segnet':
        print('Using SegNet ...')
        net = SegNet(n_channels=args.number_of_in_channels, n_classes=args.classes, bilinear=args.bilinear)
    else:
        raise NotImplementedError(
            'Please provide a valid model name: unet, deeplabv3+, fcn32s, fcn16s, fcn8s, fcns, pspnet, segnet')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)

    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=args.mask_threshold,
                           device=device,
                           target_size=args.target_size)

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
