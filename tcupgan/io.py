import numpy as np
import glob


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
            self.nch, self.d, self.h, self.w = img0.shape

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

            imgs[i, :] = data['img']/self.norm
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
                imgi = imgi/np.percentile(imgi.flatten(), 98)

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

    val_split_ind = int(ndata*val_split)
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
