import torch.nn as nn
import torch
from .lstm_layers import ConvLSTM, ConvTransposeLSTM, apply_on_channel,\
    UpSampleLSTM, time_distribute


def sample(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mu + eps*std


class LSTMVAE(nn.Module):
    def __init__(self, hidden_dims=[8, 16, 32], bottleneck_dims=[16, 8],
                 input_channels=1, final_act='elu'):

        super(LSTMVAE, self).__init__()

        self.hidden_dims = hidden_dims
        prev_filt = input_channels

        # create the encoder
        encoder_layers = []
        for i in range(len(hidden_dims) - 1):
            # each encoder layer goes from prev filter -> hidden_dims[i]
            # for each LSTM step. Then, we need to reconvolve onto the next
            # LSTM step, so we convolve to hidden_dims[i+1] filters and
            # max pool to downsample
            encoder_layers.append(ConvLSTM(prev_filt, hidden_dims[i],
                                           hidden_dims[i + 1], (3, 3), (2, 2)))

            # update the filter value for the next iteration
            prev_filt = hidden_dims[i]

        self.encoder_layers = nn.ModuleList(encoder_layers)

        prev_filt = hidden_dims[-1]

        bottleneck_layers_enc = []
        for j, filt in enumerate(bottleneck_dims):
            bottleneck_layers_enc.extend([
                nn.Conv2d(prev_filt, filt, (1, 1), padding=0,
                          stride=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(filt)])
            prev_filt = filt

        self.enc_mu = nn.Conv2d(filt, filt, (1, 1), padding=0, stride=1)
        self.enc_sig = nn.Conv2d(filt, filt, (1, 1), padding=0, stride=1)

        bottleneck_layers_dec = []
        decode_bottleneck = [*bottleneck_dims[::-1], hidden_dims[-1]]
        for j, filt in enumerate(decode_bottleneck):
            bottleneck_layers_dec.extend([
                nn.Conv2d(prev_filt, filt, (1, 1), padding=0,
                          stride=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(filt)])
            prev_filt = filt

        self.bottleneck_enc = nn.Sequential(*bottleneck_layers_enc)
        self.bottleneck_dec = nn.Sequential(*bottleneck_layers_dec)

        decoder_layers = []

        # invert the hidden layer filters for the decoder
        # also add the input channel at the end
        decoder_hidden_dims = [*hidden_dims[::-1][1:], input_channels]

        # the starting size is the last filter size of the encoder
        hidden_dim = hidden_dims[-1]
        for i in range(len(decoder_hidden_dims) - 1):
            # each decoder has the ith filter number of channels,
            # but (i+1)th filter in its cell state vector
            # in the UNet we also skip across the bottleneck, so the input is
            # the cat of both the skipped vector and the upsampling vector

            last_layer = (i == len(decoder_hidden_dims) - 2)
            decoder_layers.append(ConvTransposeLSTM(decoder_hidden_dims[i],
                                                    hidden_dim,
                                                    decoder_hidden_dims[i + 1],
                                                    (3, 3), (2, 2)))
            hidden_dim = decoder_hidden_dims[i + 1]
        
        if final_act == 'elu':
            self.pred_final = nn.ELU()
        else:
            self.pred_final = nn.Sigmoid()

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def encode(self, x):
        '''
            Create the vector embedding
        '''

        c = None
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                hidden = None
            else:
                hidden = [None, c]

            x, h, c = layer(x, hidden)

        return x, c

    def decode(self, x, c):
        '''
            Decode from the vector embedding. Note that
            this needs both the cell state and the
            bottlenecked feature
        '''
        for i, layer in enumerate(self.decoder_layers):
            x, c = layer(x, c=c)

        return x

    def forward(self, x):
        '''
            Do the encode/decode loop
        '''
        enc_x, enc_c = self.encode(x)

        # bottleneck the c dimension
        bottleneck_c = self.bottleneck_enc(enc_c)

        c_mu = self.enc_mu(bottleneck_c)
        c_sig = self.enc_sig(bottleneck_c)

        sampled_c = sample(c_mu, c_sig)

        # decode the encoded c vector for reconstruction
        dec_c = self.bottleneck_dec(sampled_c)

        img = self.decode(enc_x, dec_c)

        img = self.pred_final(img)

        return img, c_mu, c_sig


class LSTMUNet(nn.Module):
    gen_type = 'UNet'
    def __init__(self, hidden_dims=[8, 16, 32], bottleneck_dims=[16, 8],
                 input_channels=1, output_channels=4):

        super(LSTMUNet, self).__init__()

        self.hidden_dims = hidden_dims
        prev_filt = input_channels

        # create the encoder
        encoder_layers = []
        for i in range(len(hidden_dims) - 1):
            # each encoder layer goes from prev filter -> hidden_dims[i]
            # for each LSTM step. Then, we need to reconvolve onto the next
            # LSTM step, so we convolve to hidden_dims[i+1] filters and
            # max pool to downsample
            encoder_layers.append(ConvLSTM(prev_filt, hidden_dims[i],
                                           hidden_dims[i + 1], (3, 3), (2, 2)))

            # update the filter value for the next iteration
            prev_filt = hidden_dims[i]

        self.encoder_layers = nn.ModuleList(encoder_layers)

        prev_filt = hidden_dims[-1]

        bottleneck_layers_enc = []
        for j, filt in enumerate(bottleneck_dims):
            bottleneck_layers_enc.extend([
                nn.Conv3d(prev_filt, filt, (1, 1, 1), padding=0,
                          stride=1),
                nn.ReLU(True), #nn.LeakyReLU(0.2),
                nn.BatchNorm3d(filt)])
            prev_filt = filt

        bottleneck_layers_dec = []
        decode_bottleneck = [*bottleneck_dims[::-1], hidden_dims[-1]]
        for j, filt in enumerate(decode_bottleneck):
            bottleneck_layers_dec.extend([
                nn.Conv3d(prev_filt, filt, (1, 1, 1), padding=0,
                          stride=1),
                nn.ReLU(True), #nn.LeakyReLU(0.2),
                nn.BatchNorm3d(filt)])
            prev_filt = filt

        self.bottleneck_enc = nn.Sequential(*bottleneck_layers_enc)
        self.bottleneck_dec = nn.Sequential(*bottleneck_layers_dec)

        decoder_layers = []

        # invert the hidden layer filters for the decoder
        # also add the input channel at the end
        decoder_hidden_dims = [*hidden_dims[::-1][1:], output_channels]

        # the starting size is the last filter size of the encoder
        hidden_dim = hidden_dims[-1]
        for i in range(len(decoder_hidden_dims) - 1):
            # each decoder has the ith filter number of channels,
            # but (i+1)th filter in its cell state vector
            # in the UNet we also skip across the bottleneck, so the input is
            # the cat of both the skipped vector and the upsampling vector

            last_layer = (i == len(decoder_hidden_dims) - 2)
            #decoder_layers.append(ConvTransposeLSTM(decoder_hidden_dims[i] * 2,
            #                                        hidden_dim,
            #                                        decoder_hidden_dims[i + 1],
            #                                        (3, 3), (2, 2)))
            decoder_layers.append(UpSampleLSTM(decoder_hidden_dims[i] * 2,
                                                    hidden_dim,
                                                    decoder_hidden_dims[i + 1],
                                                    (3, 3), (2, 2)))
            hidden_dim = decoder_hidden_dims[i + 1]

        if output_channels > 1:
            self.pred_final = nn.Softmax(dim=2)
        else:
            self.pred_final = nn.Sigmoid()

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def encode(self, x):
        '''
            Create the vector embedding
        '''
        c = None
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                hidden = None
            else:
                hidden = [None, c]

            x, h, c = layer(x, hidden)

        return x, c

    def decode(self, x, c):
        '''
            Decode from the vector embedding. Note that
            this needs both the cell state and the
            bottlenecked feature
        '''
        for i, layer in enumerate(self.decoder_layers):
            x, c = layer(x, c)

        return x

    def forward(self, x):
        '''
            Do the encode/decode loop
        '''
        xencs = []
        cencs = []
        c = None
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                hidden = None
            else:
                hidden = [None, c]

            hidden = None

            x, c = layer(x, hidden)
            cencs.append(c)
            xencs.append(x)
            #print(c.shape)

        x = torch.swapaxes(x, 1, 2)
        # bottleneck the c dimension
        enc_x = self.bottleneck_enc(x)

        # decode the encoded c vector for reconstruction
        dec_x = self.bottleneck_dec(enc_x)
        x = torch.swapaxes(dec_x, 1, 2)

        nlayers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            # skip the x vector across the bottleneck
            xconc = torch.cat([x, xencs[nlayers - i - 1]], dim=2)
            #henc = xencs[nlayers-i-1][:,0,:,:,:]
            x = layer(xconc, c=cencs[nlayers -i -1])

        # smooth the final output mask to remove the gridding
        x = self.pred_final(x)

        return x

