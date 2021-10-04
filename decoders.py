
import torch.nn.functional as F
from torch.nn import Parameter
import torch

from numpy import arange
from numpy.random import mtrand
import numpy as np


#####################
# RNN decoder
####################

class RNN_decoder_rate2(torch.nn.Module):
    def __init__(self, args):
        super(RNN_decoder_rate2, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.dec_linear = torch.nn.Linear(args.code_rate_n * args.block_len, args.code_rate_n * args.block_len)
        self.dec_rnn = torch.nn.GRU(args.code_rate_n, args.dec_num_unit,
                                    num_layers=2, bias=True, batch_first=True,
                                    dropout=0, bidirectional=True)

        self.final = torch.nn.Linear(2 * args.dec_num_unit, 1)

    def set_parallel(self):
        pass

    def forward(self, received):

        received = received.type(torch.FloatTensor).to(self.this_device) #(batch_size, n*block_len,1)
        permuted = received.permute(0, 2, 1) #(batch_size, 1, n*block_len)
        code = self.dec_linear(permuted) #(batch_size, 1, n*block_len)
        code= code.view(self.args.batch_size,self.args.block_len,self.args.code_rate_n) #(batch_size,block_len,n) [ab,ab]

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
        self.dec_linear = torch.nn.Linear(args.code_rate_n * args.block_len, 2* args.code_rate_k * args.block_len)

        self.dec_rnn = torch.nn.GRU(args.code_rate_n, args.dec_num_unit,
                                    num_layers=2, bias=True, batch_first=True,
                                    dropout=0, bidirectional=True)

        self.final = torch.nn.Linear(2 * args.dec_num_unit, args.code_rate_k)

    def set_parallel(self):
        pass

    def forward(self, received):

        received = received.type(torch.FloatTensor).to(self.this_device) #(batch_size, n*block_len,1)
        permuted = received.permute(0, 2, 1) #(batch_size, 1, n*block_len)
        code = self.dec_linear(permuted) #(batch_size, 1, n*block_len)
        code= code.view(self.args.batch_size,self.args.block_len, self.args.code_rate_n) #(batch_size,block_len,n) [ab,ab]

        #  Decoder
        x_plr      = self.dec_rnn(code)[0]
        final      = torch.sigmoid(self.final(x_plr))
        return final
