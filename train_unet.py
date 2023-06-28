from tcupgan import LSTMUNet, PatchDiscriminator, create_generators
from tcupgan.trainer import Trainer
from torchinfo import summary

device = 'cuda'

base_folder = '/home/fortson/rsankar/data/BraTS2021/training.resized/'
train_data, val_data = create_generators(base_folder,
                                         batch_size=12)

hidden = [6, 12, 18, 64, 128]
generator = LSTMUNet(hidden_dims=hidden,
                     input_channels=4,
                     output_channels=4).to(device)

discriminator = PatchDiscriminator(input_channels=8, nlayers=2, nfilt=16).to(device)

summary(generator, input_size=(1, 155, 4, 128, 128), device=device)
summary(discriminator, input_size=(1, 155, 8, 128, 128), device=device)


trainer = Trainer(generator, discriminator, 'checkpoints.resized-gamma05/')
# trainer.load_last_checkpoint()

gen_learning_rate = 1.e-3
dsc_learning_rate = 1.e-4

trainer.seg_alpha = 100
trainer.loss_type = 'tversky'

trainer.train(train_data, val_data, 150, lr_decay=0.95,
              dsc_learning_rate=dsc_learning_rate,
              gen_learning_rate=gen_learning_rate)
