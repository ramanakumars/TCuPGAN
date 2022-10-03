from tcupgan import LSTMVAE, PatchDiscriminator
from tcupgan.io import create_generators
from tcupgan.trainer import Trainer
from torchinfo import summary
import torch
import tqdm
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR


device = 'cuda'

base_folder = '/home/fortson/manth145/data/MANGA_EmLine_Cubes/'
train_data, val_data = create_generators(base_folder,
                                         batch_size=12, datatype='npy')


hidden = [8, 12, 18, 24]
bottleneck_dims = [64, 32, 16, 8, 6]
generator = LSTMVAE(hidden_dims=hidden, bottleneck_dims=bottleneck_dims,
                     input_channels=1).to(device)

discriminator = PatchDiscriminator(input_channels=2, nfilt=8, nlayers=1).to(device)

summary(discriminator, input_size=(1, 190, 2, 32, 32), device=device)


trainer = Trainer(generator, discriminator, 'checkpoints.MaNGA/')
trainer.disc_output = [23, 4, 4]
trainer.kl_beta = 0.1
#trainer.load_last_checkpoint()

gen_learning_rate = 1.e-3*(0.95)**((trainer.start-1)/5)
dsc_learning_rate = 1.e-4*(0.95)**((trainer.start-1)/5)

trainer.train(train_data, val_data, 150, lr_decay=0.95, 
              dsc_learning_rate=dsc_learning_rate,
              gen_learning_rate=gen_learning_rate)

