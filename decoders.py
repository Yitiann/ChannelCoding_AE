
import torch.nn.functional as F
from torch.nn import Parameter
import torch

from numpy import arange
from numpy.random import mtrand
import numpy as np


##################################################
# Rate 1/2 CNN
##################################################

class DEC_LargeCNN_rate2(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_LargeCNN_rate2, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_cnns.append(SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )

            self.dec2_cnns.append(SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )
            self.dec1_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_cnns[idx] = torch.nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = torch.nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)
        self.deinterleaver.set_parray(p_array)

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys       = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_sys_int   = self.interleaver(r_sys)
        r_par       = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        r_par_deint = self.deinterleaver(r_par)

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys,r_par_deint, prior], dim = 2)

            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par, x_plr_int ], dim = 2)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par_deint, prior], dim = 2)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par, x_plr_int ], dim = 2)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))

        return final

#####################
# RNN decoder
####################

class RNN_decoder_rate2(torch.nn.Module):
    def __init__(self, args):
        super(RNN_decoder_rate2, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        #self.dec_linear = torch.nn.Linear(args.code_rate_n * args.block_len, args.code_rate_n * args.block_len)                                   
        self.dec_rnn = torch.nn.GRU(args.code_rate_n, args.dec_num_unit,
                                    num_layers=2, bias=True, batch_first=True,
                                    dropout=0, bidirectional=True)

        self.final = torch.nn.Linear(2 * args.dec_num_unit, 1)

    def set_parallel(self):
        pass

    def forward(self, received):
        # batch_size, _, sequence_length = received.shape
        # noised_code = torch.zeros(size=[batch_size,self.args.code_rate_n, sequence_length / self.args.code_rate_n]).to("cuda")

        # noised_code[:,0,:] = received[:, :, :sequence_length - 1]
        # noised_code[:,1,:] = received[:, :,sequence_length]
        received = received.type(torch.FloatTensor).to(self.this_device) #(batch_size, n*block_len,1)
        permuted = received.permute(0, 2, 1) #(batch_size, 1, n*block_len)
        #code = self.dec_linear(permuted) #(batch_size, 1, n*block_len)
        code= permuted.view(self.args.batch_size,self.args.block_len,self.args.code_rate_n) #(batch_size,block_len,n) [ab,ab]

        #  Decoder
        x_plr  = self.dec_rnn(code)[0]
        final  = torch.sigmoid(self.final(x_plr))
        return final

class RNN_decoder_rate_high(torch.nn.Module):
    def __init__(self, args):
        super(RNN_decoder_rate_high, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        #self.dec_linear = torch.nn.Linear(args.code_rate_n * args.block_len, 2* args.code_rate_k * args.block_len)

        self.dec_rnn = torch.nn.GRU(args.code_rate_n, args.dec_num_unit,
                                    num_layers=2, bias=True, batch_first=True,
                                    dropout=0, bidirectional=True)

        self.final = torch.nn.Linear(2 * args.dec_num_unit, args.code_rate_k)

    def set_parallel(self):
        pass

    def forward(self, received):

        received = received.type(torch.FloatTensor).to(self.this_device) #(batch_size, n*block_len,1)
        permuted = received.permute(0, 2, 1) #(batch_size, 1, n*block_len)
        #code = self.dec_linear(permuted) #(batch_size, 1, n*block_len)
        code= permuted.view(self.args.batch_size,self.args.block_len, self.args.code_rate_n) #(batch_size,block_len,n) [ab,ab]

        #  Decoder
        x_plr      = self.dec_rnn(code)[0]
        final      = torch.sigmoid(self.final(x_plr))
        return final
