
import torch
from utils import snr_db2sigma, snr_sigma2db
import numpy as np

def generate_noise(noise_shape, args, test_sigma = 'default', snr_low = 0.0, snr_high = 0.0, mode = 'encoder'):
    # SNRs at training
    if test_sigma == 'default':
        if args.channel == 'bec':
            if mode == 'encoder':
                this_sigma = args.bec_p_enc
            else:
                this_sigma = args.bec_p_dec

        elif args.channel in ['bsc', 'ge']:
            if mode == 'encoder':
                this_sigma = args.bsc_p_enc
            else:
                this_sigma = args.bsc_p_dec
        else: # general AWGN cases
            this_sigma_low = snr_db2sigma(snr_low)
            this_sigma_high= snr_db2sigma(snr_high)
            # mixture of noise sigma.
            this_sigma = (this_sigma_low - this_sigma_high) * torch.rand(noise_shape) + this_sigma_high

    else:
        if args.channel in ['bec', 'bsc', 'ge']:  # bsc/bec noises
            this_sigma = test_sigma
        else:
            this_sigma = snr_db2sigma(test_sigma)

        # Unspecific channel, use AWGN channel.
    fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    return fwd_noise

def ISI(x,device):
    '''
    implement the linear inter-symbols intervention
    :param x:input clear codeword with shape [batchsize, L*n, 1 ], 2 means its a complex sequence
    :return:
    '''

    batch_size, sequence_length, _ = x.shape

    h=torch.tensor([0.5,0.5,-0.5,-0.5])
    #h = torch.tensor([0.3482, 0.8704, 0.348]) # channel power delay profile
    # x_real = x[:, 0, :]
    # x_image = x[:, 1, :]
    #out = torch.stack([out_real, out_image], dim=2)


    x_delay_1 = torch.zeros(size=x.shape).to(device)
    x_delay_2 = torch.zeros(size=x.shape).to(device)
    x_delay_3 = torch.zeros(size=x.shape).to(device)

    x_delay_1[:, 1:,0] = x[:, :sequence_length-1,0]
    x_delay_2[:, 2:,0] = x[:, :sequence_length-2,0]
    x_delay_3[:, 3:,0] = x[:, :sequence_length-3,0]

    # linear convolution of the input codeword and channel impluse response,
    out = h[0] * x + h[1] * x_delay_1 + h[2] * x_delay_2 + h[3] * x_delay_3

    return out
