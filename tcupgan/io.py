import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torch.nn.functional import one_hot
from torchvision import transforms
from einops import rearrange
import os
import tqdm
import signal


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class DataGenerator:
    file_type = 'npz'

    def __init__(self, datafolder,
                 batch_size, inchannels=3, outchannels=3,
                 indices=None, norm=1.):
        self.img_datafolder = datafolder
        self.batch_size = batch_size

        self.imgfiles = np.asarray(
            sorted(glob.glob(datafolder + f"*.{self.file_type}")))

        if indices is None:
            self.indices = np.arange(len(self.imgfiles))
        else:
            self.indices = indices

        self.ndata = len(self.indices)

        self.inchannels = inchannels
        self.outchannels = outchannels

        self.norm = norm
        self.perc_normalize = False

        # get info about the data

        if self.file_type == 'npz':
            img0 = np.load(self.imgfiles[0])['img']
        else:
            img0 = np.load(self.imgfiles[0])

        if len(img0.shape) == 3:
            self.d, self.h, self.w = img0.shape
        else:
            self.d, self.nch, self.h, self.w = img0.shape

        print(f"Found {self.ndata} images of shape {self.w}x{self.h}x{self.d} with {self.nch} channels")

    def __len__(self):
        return self.ndata // self.batch_size

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        if index > len(self):
            raise StopIteration

        batch_indices = self.indices[index * self.batch_size:
                                     (index + 1) * self.batch_size]
        data = self.get_from_indices(batch_indices)
        return data

    def get_from_indices(self, batch_indices):
        imgfiles = self.imgfiles[batch_indices]

        imgs = np.zeros((len(imgfiles), self.d, self.inchannels, self.h, self.w))
        segs = np.zeros((len(imgfiles), self.d, self.outchannels, self.h, self.w))

        for i in range(len(imgfiles)):
            data = np.load(imgfiles[i])

            imgs[i, :] = data['img'] / self.norm
            segs[i, :] = data['mask']

        return imgs, segs


class NpyDataGenerator(DataGenerator):
    file_type = 'npy'

    def get_from_indices(self, batch_indices):
        imgfiles = self.imgfiles[batch_indices]

        imgs = np.zeros((len(imgfiles), self.d, self.nch, self.h, self.w))

        for i, imgi in enumerate(imgfiles):
            imgi = np.load(imgi)
            if self.perc_normalize:
                imgi = imgi / np.percentile(imgi.flatten(), 98)

            imgs[i, :] = np.transpose(imgi, (1, 0, 2, 3))

        return imgs, None


def create_generators(img_datafolder, batch_size, inchannels=3,
                      outchannels=3, val_split=0.1, datatype='npz',
                      norm=1.):
    if datatype == 'npz':
        imgfiles = np.asarray(sorted(glob.glob(img_datafolder + "*.npz")))
    elif datatype == 'npy':
        imgfiles = np.asarray(sorted(glob.glob(img_datafolder + "*.npy")))

    ndata = len(imgfiles)

    print(f"Loading data with {ndata} images")

    inds = np.arange(ndata)
    np.random.shuffle(inds)

    val_split_ind = int(ndata * val_split)
    val_ind = inds[:val_split_ind]
    training_ind = inds[val_split_ind:]

    if datatype == 'npz':
        train_data = DataGenerator(img_datafolder, batch_size,
                                   indices=training_ind, inchannels=inchannels,
                                   outchannels=outchannels, norm=norm)
        val_data = DataGenerator(img_datafolder, batch_size,
                                 indices=val_ind, inchannels=inchannels,
                                 outchannels=outchannels, norm=norm)
    elif datatype == 'npy':
        train_data = NpyDataGenerator(img_datafolder, batch_size,
                                      indices=training_ind, inchannels=inchannels,
                                      outchannels=outchannels, norm=norm)
        val_data = NpyDataGenerator(img_datafolder, batch_size,
                                    indices=val_ind, inchannels=inchannels,
                                    outchannels=outchannels, norm=norm)

    return train_data, val_data


class VideoDataGenerator(Dataset):
    size = 192

    def __init__(self, root_folder, max_frames=10, in_channels=3, out_channels=126, verbose=False):
        self.root_folder = root_folder
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.frames = []
        self.data = []
        for name in tqdm.tqdm(sorted(os.listdir(self.root_folder)), desc='Building dataset', disable=not verbose, ascii=True):
            folder = os.path.join(self.root_folder, name)
            if not os.path.isdir(folder):
                continue
            frames = sorted(glob.glob(os.path.join(folder, "origin/*.jpg")))
            maskframes = sorted(glob.glob(os.path.join(folder, "mask/*.png")))

            if len(maskframes) != len(frames):
                continue

            nframes = len(frames)

            assert max_frames <= nframes,\
                f"Maximum slice dimension is greater than the number of frames in {folder}"

            nbatches = nframes // max_frames + 1

            self.frames.append(nframes)

            for i in range(nbatches):
                # store the data as the folder and the start frame
                self.data.append([folder, min([i * max_frames, nframes - max_frames])])

        self.transform = transforms.Resize((self.size, self.size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        folder, start = self.data[index]

        img_fnames = sorted(glob.glob(os.path.join(self.root_folder, f"{folder}/origin/*.jpg")))[start:start + self.max_frames]

        imgs = torch.zeros((self.max_frames, self.in_channels, self.size, self.size))
        masks = torch.zeros((self.max_frames, self.out_channels, self.size, self.size))

        for i, fname in enumerate(img_fnames):
            img, mask = self.get_image_mask_pair(fname)
            imgs[i, :] = self.transform(img)
            masks[i, :] = self.transform(mask)

        return imgs, masks

    def get_image_mask_pair(self, fname):
        img = read_image(fname, ImageReadMode.RGB)
        mask = read_image(fname.replace("origin", "mask").replace('.jpg', '.png'), ImageReadMode.GRAY)[0, :]

        mask[mask > 124] = 0
        mask = one_hot(
            mask.to(torch.int64),
            num_classes=self.out_channels
        )
        mask = rearrange(mask, "h w c -> c h w")

        return img, mask


class NpzDataSet(Dataset):
    file_type = 'npz'

    def __init__(self, datafolder, inchannels=3, outchannels=3, norm=1.):
        self.img_datafolder = datafolder

        self.imgfiles = np.asarray(
            sorted(glob.glob(datafolder + f"*.{self.file_type}")))

        self.indices = np.arange(len(self.imgfiles))

        self.ndata = len(self.indices)

        self.inchannels = inchannels
        self.outchannels = outchannels

        self.norm = norm
        self.perc_normalize = False

        # get info about the data

        if self.file_type == 'npz':
            img0 = np.load(self.imgfiles[0])['img']
        else:
            img0 = np.load(self.imgfiles[0])

        if len(img0.shape) == 3:
            self.d, self.h, self.w = img0.shape
        else:
            self.d, self.nch, self.h, self.w = img0.shape

        print(f"Found {self.ndata} images of shape {self.w}x{self.h}x{self.d} with {self.nch} channels")

    def __len__(self):
        return self.ndata

    def __getitem__(self, index):
        imgfile = self.imgfiles[index]

        data = np.load(imgfile)

        imgs = torch.as_tensor(data['img'], dtype=torch.float) / self.norm
        segs = torch.as_tensor(data['mask'], dtype=torch.float)

        return imgs, segs
