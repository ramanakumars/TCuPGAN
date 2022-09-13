from tcupgan import LSTMUNet, PatchDiscriminator, create_generators
from tcupgan.losses import fc_tversky
from tcupgan.trainer import Trainer
from torchinfo import summary
import torch
import tqdm
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR


device = 'cuda'

base_folder = '/home/fortson/manth145/data/BraTS2020/Processed/Unaugmented/Training/'
train_data, val_data = create_generators(base_folder + 'flair/', 
                                        base_folder + 'seg/', batch_size=8)


hidden        = [8, 16, 32, 64, 128]
generator = LSTMUNet(hidden_dims=hidden).to(device)

discriminator = PatchDiscriminator(input_channels=5, nfilt=8) 

summary(discriminator, input_size=(1, 155, 5, 128, 128), device=device)


trainer = Trainer(generator, discriminator, 'checkpoints-gamma05/')
trainer.fc_beta = 0.7

trainer.load_last_checkpoint()

trainer.train(train_data, val_data, 150, lr_decay=0.95)
