import numpy as np
import glob


class DataGenerator:
    file_type = 'npz'

    def __init__(self, datafolder,
                 batch_size, indices=None):
        self.img_datafolder = datafolder
        self.batch_size = batch_size

        self.imgfiles = np.asarray(
            sorted(glob.glob(datafolder + f"*.{self.file_type}")))

        if indices is None:
            self.indices = np.arange(len(self.imgfiles))
        else:
            self.indices = indices

        self.ndata = len(self.indices)

        # get info about the data

        if self.file_type == '.npz':
            img0 = np.load(self.imgfiles[0])['img']
        else:
            img0 = np.load(self.imgfiles[0])

        if len(img0.shape) == 3:
            self.d, self.h, self.w = img0.shape
        else:
            self.d, _, self.h, self.w = img0.shape

        print(f"Found {self.ndata} images of shape {self.w}x{self.h}x{self.d}")

    def __len__(self):
        return self.ndata // self.batch_size

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:
                                     (index + 1) * self.batch_size]

        imgfiles = self.imgfiles[batch_indices]

        imgs = np.zeros((self.batch_size, self.d, 4, self.h, self.w))
        segs = np.zeros((self.batch_size, self.d, 4, self.h, self.w))

        for i in range(self.batch_size):
            data = np.load(imgfiles[i])

            imgs[i, :] = data['img']
            segs[i, :] = data['mask']

        return imgs, segs


class NpyDataGenerator(DataGenerator):
    file_type = 'npy'

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:
                                     (index + 1) * self.batch_size]

        imgfiles = self.imgfiles[batch_indices]

        imgs = np.zeros((self.batch_size, self.d, 1, self.h, self.w))

        for i in range(self.batch_size):
            imgs[i, :, 0, :, :] = np.load(imgfiles[i])

        return imgs, None


def create_generators(img_datafolder, batch_size,
                      val_split=0.1, datatype='npz'):
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
                                   indices=training_ind)
        val_data = DataGenerator(img_datafolder, batch_size,
                                 indices=val_ind)
    elif datatype == 'npy':
        train_data = NpyDataGenerator(img_datafolder, batch_size,
                                      indices=training_ind)
        val_data = NpyDataGenerator(img_datafolder, batch_size,
                                    indices=val_ind)

    return train_data, val_data
