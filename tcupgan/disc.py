import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    '''
        Patch based discriminator which predicts
        the probability whether a subset cube of the image
        is real or fake.
    '''

    def __init__(self, input_channels, nlayers=3, nfilt=16, dropout=0.5):
        super(PatchDiscriminator, self).__init__()

        # the first convolution goes from the input channels
        # to the first number of filters
        prev_filt = input_channels
        next_filt = nfilt

        layers = []
        for i in range(nlayers + 1):
            # for each layer, we apply conv, act and normalization
            layers.append(nn.Conv3d(prev_filt, next_filt, (4, 4, 4),
                                    stride=(2, 2, 2),
                                    padding=(1, 1, 1), bias=False))
            layers.append(nn.LeakyReLU(0.2, True))
            layers.append(nn.BatchNorm3d(next_filt))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # the number of filters exponentially increase
            prev_filt = next_filt
            next_filt = next_filt * min([2**i, 8])

        # last predictive layer
        layers.append(nn.Conv3d(prev_filt, 1, (4, 4, 4), stride=(2, 2, 2),
                                padding=(1, 1, 1), bias=False))
        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        # switch channel and time axis before doing the 3d conv
        return self.discriminator(torch.swapaxes(x, 1, 2))
