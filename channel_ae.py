__author__ = 'yihanjiang'
import torch
import torch.nn.functional as F
import numpy as np
from channels import ISI

class Channel_AE(torch.nn.Module):
    def __init__(self, args, enc, dec):
        super(Channel_AE, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec

    def forward(self, input, fwd_noise):
        if self.args.encoder == 'conv':
            codes  = self.enc.conv_enc(input)
        else:
            codes = self.enc(input)

        # Setup channel mode:
        if self.args.channel == 'awgn':
            received_codes = codes + fwd_noise

        elif self.args.channel == 'fading':
            data_shape = codes.shape
            #  Rayleigh Fading Channel, non-coherent
            fading_h = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) #np.sqrt(2.0)
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)
            received_codes = fading_h*codes + fwd_noise

            # fading_h = np.sqrt(np.random.standard_normal(data_shape)**2 +  np.random.standard_normal(data_shape)**2)/np.sqrt(3.14/2.0)
            # noise = sigma * np.random.standard_normal(data_shape) # Define noise
            #
            # # corrupted_signal = 2.0*fading_h*input_signal-1.0 + noise
            # corrupted_signal = fading_h *(2.0*input_signal-1.0) + noise
        elif self.args.channel == 'isi':
            received_codes = ISI(codes,self.this_device) + fwd_noise
        else:
            print('default AWGN channel')
            received_codes = codes + fwd_noise

        x_dec  = self.dec(received_codes)

        return x_dec, codes



