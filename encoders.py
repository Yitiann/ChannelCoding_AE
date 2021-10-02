'''
This module contains all possible encoders, STE, and utilities.
'''

import torch
import torch.nn.functional as F
import commpy.channelcoding.convcode as cc
from numpy import arange
from numpy.random import mtrand
import math
import numpy as np

from utils import snr_db2sigma

##############################################
# STE implementation
##############################################

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, args):

        ctx.save_for_backward(inputs)
        ctx.args = args

        x_lim_abs  = args.enc_value_limit
        x_lim_range = 2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if args.enc_quantize_level == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((args.enc_quantize_level - 1.0)/x_lim_range)) * x_lim_range/(args.enc_quantize_level - 1.0) - x_lim_abs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.args.enc_clipping in ['inputs', 'both']:
            input, = ctx.saved_tensors
            grad_output[input>ctx.args.enc_value_limit]=0
            grad_output[input<-ctx.args.enc_value_limit]=0

        if ctx.args.enc_clipping in ['gradient', 'both']:
            grad_output = torch.clamp(grad_output, -ctx.args.enc_grad_limit, ctx.args.enc_grad_limit)

        if ctx.args.train_channel_mode not in ['group_norm_noisy', 'group_norm_noisy_quantize']:
            grad_input = grad_output.clone()
        else:
            # Experimental pass gradient noise to encoder.
            grad_noise = snr_db2sigma(ctx.args.fb_noise_snr) * torch.randn(grad_output[0].shape, dtype=torch.float)
            ave_temp   = grad_output.mean(dim=0) + grad_noise
            ave_grad   = torch.stack([ave_temp for _ in range(ctx.args.batch_size)], dim=2).permute(2,0,1)
            grad_input = ave_grad + grad_noise

        return grad_input, None


##############################################
# Encoder Base.
# Power Normalization is implemented here.
##############################################
class ENCBase(torch.nn.Module):
    def __init__(self, args):
        super(ENCBase, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.reset_precomp()

    def set_parallel(self):
        pass

    def set_precomp(self, mean_scalar, std_scalar):
        self.mean_scalar = mean_scalar.to(self.this_device)
        self.std_scalar  = std_scalar.to(self.this_device)

    # not tested yet
    def reset_precomp(self):
        self.mean_scalar = torch.zeros(1).type(torch.FloatTensor).to(self.this_device)
        self.std_scalar  = torch.ones(1).type(torch.FloatTensor).to(self.this_device)
        self.num_test_block= 0.0

    def enc_act(self, inputs):
        if self.args.enc_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.enc_act == 'elu':
            return F.elu(inputs)
        elif self.args.enc_act == 'relu':
            return F.relu(inputs)
        elif self.args.enc_act == 'selu':
            return F.selu(inputs)
        elif self.args.enc_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.enc_act == 'linear':
            return inputs
        else:
            return inputs

    def power_constraint(self, x_input):

        if self.args.no_code_norm:
            return x_input
        else:
            this_mean    = torch.mean(x_input)
            this_std     = torch.std(x_input)

            if self.args.precompute_norm_stats:
                self.num_test_block += 1.0
                self.mean_scalar = (self.mean_scalar*(self.num_test_block-1) + this_mean)/self.num_test_block
                self.std_scalar  = (self.std_scalar*(self.num_test_block-1) + this_std)/self.num_test_block
                x_input_norm = (x_input - self.mean_scalar)/self.std_scalar
            else:
                x_input_norm = (x_input-this_mean)*1.0 / this_std
            
            if self.args.train_channel_mode == 'block_norm_ste':
                stequantize = STEQuantize.apply
                x_input_norm = stequantize(x_input_norm, self.args)

            return x_input_norm



###################################
# Basic RNN AE model v0.0
#    rate 1/2 fixed
#######################################################
class RNN_encoder_rate2(ENCBase):
    def __init__(self, args):
        # only for code rate 1/2
        super(RNN_encoder_rate2, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_rnn_1 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)

        self.enc_linear_1 = torch.nn.Linear(2 * args.enc_num_unit, 1)

        self.enc_rnn_2 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)

        self.enc_linear_2 = torch.nn.Linear(2 * args.enc_num_unit, 1)
        #self.enc_linear_final = torch.nn.Linear(2*args.block_len,2*args.block_len)

    def set_parallel(self):
        self.enc_rnn_1 = torch.nn.DataParallel(self.enc_rnn_1)
        self.enc_rnn_2 = torch.nn.DataParallel(self.enc_rnn_2)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)

    def forward(self, inputs):
        x_sys      = self.enc_rnn_1(inputs)[0]
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_rnn_2(inputs)[0]
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_tx       = torch.cat([x_sys,x_p1], dim = 2) #(batch_size,block_length,2) [a,b
                                                                                  # a,b...]
        x_tx = x_tx.view(self.args.batch_size,1,-1)   #(bs, 1, 2*block_length)[abab...]

        #x_tx = self.enc_linear_final(x_tx)
        permuted = x_tx.permute(0, 2, 1) #(bs,2*block_length,1)

        codes = self.power_constraint(permuted)
        #codes = F.tanh(permuted)

        return codes

##################################
# Basic RNN AE model v0.0
#    rate 1/4 fixed
#######################################################
class RNN_encoder_rate4(ENCBase):
    def __init__(self, args):
        # only for code rate 1/4
        super(RNN_encoder_rate4, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_rnn_1 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)

        self.enc_linear_1 = torch.nn.Linear(2 * args.enc_num_unit, 1)

        self.enc_rnn_2 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)
        self.enc_linear_2 = torch.nn.Linear(2 * args.enc_num_unit, 1)

        self.enc_rnn_3 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)
        self.enc_linear_3 = torch.nn.Linear(2 * args.enc_num_unit, 1)
        self.enc_rnn_4 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)
        self.enc_linear_4 = torch.nn.Linear(2 * args.enc_num_unit, 1)
        #self.enc_linear_final = torch.nn.Linear(4*args.block_len,4*args.block_len)
                                                

    def set_parallel(self):
        self.enc_rnn_1 = torch.nn.DataParallel(self.enc_rnn_1)
        self.enc_rnn_2 = torch.nn.DataParallel(self.enc_rnn_2)
        self.enc_rnn_3 = torch.nn.DataParallel(self.enc_rnn_3)
        self.enc_rnn_4 = torch.nn.DataParallel(self.enc_rnn_4)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)
        self.enc_linear_4 = torch.nn.DataParallel(self.enc_linear_4)
    def forward(self, inputs):
        x_sys      = self.enc_rnn_1(inputs)[0]
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_rnn_2(inputs)[0]
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_p2       = self.enc_rnn_3(inputs)[0]
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_p3       = self.enc_rnn_4(inputs)[0]
        x_p3       = self.enc_act(self.enc_linear_4(x_p3))

        x_tx       = torch.cat([x_sys,x_p1,x_p2,x_p3], dim = 2)
        x_tx = x_tx.view(self.args.batch_size,1,-1)   #(bs, 1, 4*block_length)[abcdabcd...]

        #x_tx = self.enc_linear_final(x_tx)
        permuted = x_tx.permute(0, 2, 1) #(bs,4*block_length,1)

        codes = self.power_constraint(permuted)

        return codes


#################################
# Basic RNN AE model v0.0
#    rate 1/3 fixed
#######################################################
class RNN_encoder_rate3(ENCBase):
    def __init__(self, args):
        # only for code rate 1/3
        super(RNN_encoder_rate3, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_rnn_1 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)

        self.enc_linear_1 = torch.nn.Linear(2 * args.enc_num_unit, 1)

        self.enc_rnn_2 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)
        self.enc_linear_2 = torch.nn.Linear(2 * args.enc_num_unit, 1)

        self.enc_rnn_3 = torch.nn.GRU(1, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)
        self.enc_linear_3 = torch.nn.Linear(2 * args.enc_num_unit, 1)
        #self.enc_linear_final = torch.nn.Linear(3*args.block_len,3*args.block_len)
                                                

    def set_parallel(self):
        self.enc_rnn_1 = torch.nn.DataParallel(self.enc_rnn_1)
        self.enc_rnn_2 = torch.nn.DataParallel(self.enc_rnn_2)
        self.enc_rnn_3 = torch.nn.DataParallel(self.enc_rnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs):
        x_sys      = self.enc_rnn_1(inputs)[0]
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_rnn_2(inputs)[0]
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_p2       = self.enc_rnn_3(inputs)[0]
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1,x_p2], dim = 2)
        x_tx = x_tx.view(self.args.batch_size,1,-1)   #(bs, 1, 3*block_length)[abcabc...]

        #x_tx = self.enc_linear_final(x_tx)
        permuted = x_tx.permute(0, 2, 1) #(bs,3*block_length,1)

        codes = self.power_constraint(permuted)

        return codes

###################################
# Basic RNN AE model v0.0
#    rate >1/2
#######################################################
class RNN_encoder_rate_high(ENCBase):
    def __init__(self, args):
        super(RNN_encoder_rate_high, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_rnn_1 = torch.nn.GRU(args.code_rate_k, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)

        self.enc_linear_1 = torch.nn.Linear(2 * args.enc_num_unit, args.code_rate_k)

        self.enc_rnn_2 = torch.nn.GRU(args.code_rate_k, args.enc_num_unit,
                                       num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                       dropout=0, bidirectional=True)

        self.enc_linear_2 = torch.nn.Linear(2 * args.enc_num_unit, args.code_rate_k)
        self.enc_linear_final = torch.nn.Linear(2*args.code_rate_k * args.block_len, args.code_rate_n * args.block_len
                                                )


    def set_parallel(self):
        self.enc_rnn_1 = torch.nn.DataParallel(self.enc_rnn_1)
        self.enc_rnn_2 = torch.nn.DataParallel(self.enc_rnn_2)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)

    def forward(self, inputs):
        x_sys      = self.enc_rnn_1(inputs)[0]
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_rnn_2(inputs)[0]
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_tx       = torch.cat([x_sys,x_p1], dim = 2)#(batch_size,block_length,4) [a1,a2,b1,b2
                                                                                  # a1,a2,b1,b2...]
        x_tx = x_tx.view(self.args.batch_size,1,-1)   #(bs, 1, 2*k*block_length)[a1a2b1b2 a1a2b1b2...]

        # codes_before_nopwr = x_tx
        #codes_before = F.tanh(x_tx)

        x_tx = self.enc_linear_final(x_tx)
        permuted = x_tx.permute(0, 2, 1) #(bs,n*block_length,1)

        b1=(permuted<0) & (permuted>-2)
        b2=(permuted>=0) & (permuted<2)
        b3 = (permuted<=-2) | (permuted>=2)
        a1 = permuted-1000
        a2 = permuted+1000
        a3 = permuted

        code = a1*b1 + a2*b2 + a3 * b3

        #codes_after_nopwr = permuted
        codes_after = torch.tanh(code)
        #codes_after = (F.tanh(permuted)+1)/2
        #codes_after = torch.sign(permuted)
        #codes_after = (torch.sign(permuted)+1)/2
        #codes_after = self.power_constraint(permuted)

        # #print("codes before dense 2, no power normalize:", codes_before_nopwr)
        # print("codes before dense 2:", codes_before)
        # # # print("codes after dense 2, no power normalize:", codes_after_nopwr)
        # test=codes_after.permute(0, 2, 1)
        # print("codes after dense 2:", test)

        return codes_after

class traditional_enc(ENCBase):
    def __init__(self, args):
        super(traditional_enc, self).__init__(args)
        self.args = args
    def conv_enc(self,inputs):
        num_block = self.args.batch_size
        block_len = self.args.block_len
        x_code    = []

        #generator_matrix = np.array([[561, 753]])
        generator_matrix=np.array([[7,5]])
        M = np.array([2]) # Number of delay elements in the convolutional encoder
        trellis = cc.Trellis(M, generator_matrix)# Create trellis data structure
        #puncpat=np.array([1,1,1,1,0,0])
        #puncpat=np.array([1,1,0,1,1,0,1,0])
        puncpat=np.array([1,1,1,0])

        puncpat=puncpat.reshape(1,4)
        inputs=inputs.reshape(num_block, self.args.code_rate_k * block_len, 1)
        #x=inputs.numpy()
        # X_train_raw = np.random.randint(0, 2, block_len * num_block*2)
        # X_train_raw = X_train_raw.reshape((num_block,2*block_len, 1))

        for idx in range(num_block):
            xx = cc.conv_encode(inputs[idx, :, 0], trellis, termination='cont',puncture_matrix=puncpat)
            #xx = cc.conv_encode(inputs[idx, :, 0], trellis, termination='cont')
            #xx = cc.conv_encode(inputs[idx,:,0],trellis)

            xx = xx[:-block_len]
            xx = xx.reshape((block_len, self.args.code_rate_n))

            x_code.append(xx)
        code = torch.FloatTensor(x_code)
        x_code=code.reshape(num_block, self.args.code_rate_n * block_len,1)

        modulated =x_code*2-1

        modulated = modulated.to(self.this_device)
        return modulated