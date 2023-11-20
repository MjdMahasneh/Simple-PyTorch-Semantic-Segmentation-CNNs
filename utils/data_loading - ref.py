import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', target_size : tuple = (None, None), stage: str = 'unknown'):

        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.target_size = target_size ## (height, width)


        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.ids = [p.relative_to(self.images_dir).as_posix()[:-4] for p in self.images_dir.rglob('*.jpg')] ## as_posix() converts \ to / in windows
        # ## collect all files ending with .jpg using os.walk
        # self.ids_2 = []
        # for root, dirs, files in os.walk(images_dir):
        #     for file in files:
        #         if file.endswith(".jpg"):
        #              # self.ids.append(os.path.join(root, file))
        #              ## append relative path to image_dir
        #              self.ids_2.append(os.path.relpath(os.path.join(root, file), images_dir))
        self.ids = list(set(self.ids))

        ## sort and check if ids and ids_2 are the same
        # self.ids.sort()
        # self.ids_2.sort()
        #
        # print('self.ids:', self.ids)
        # print('-'*50)
        # print('self.ids_2:', self.ids_2)
        #
        # assert self.ids == self.ids_2, 'ids and ids_2 are not the same'
        #



        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')


        # raise Exception('stop here!!')



        logging.info(f'Creating {stage} dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values in {stage} dataset')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values in {stage} dataset : {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    # def preprocess(mask_values, pil_img, scale, is_mask, target_h=512, target_w=512):
    # def preprocess(mask_values, pil_img, scale, is_mask):
    def preprocess(mask_values, pil_img, is_mask, target_h=512, target_w=512):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        assert target_h > 0 and target_w > 0, 'Scale is too small, resized images would have no pixel'
        assert target_h != None and target_w != None, 'target_h and target_w must be specified'

        pil_img = pil_img.resize((target_w, target_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((target_h, target_w), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, target_h=self.target_size[0], target_w=self.target_size[1])
        # mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, target_h=self.target_size[0], target_w=self.target_size[1])
        img = self.preprocess(self.mask_values, img, is_mask=False, target_h=self.target_size[0], target_w=self.target_size[1])
        mask = self.preprocess(self.mask_values, mask, is_mask=True, target_h=self.target_size[0], target_w=self.target_size[1])

        # # visualize image and mask side by side
        # plt.subplot(1, 2, 1)
        # # plt.imshow(img[1][:, :])
        # plt.imshow(img.transpose(1, 2, 0))
        # plt.title('img')
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask)
        # plt.title('mask')
        # plt.axis('off')
        # plt.show()

        # raise Exception('stop here!!')



        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):

    '''Carvana dataset class for loading images and masks inherits from BasicDataset'''

    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
