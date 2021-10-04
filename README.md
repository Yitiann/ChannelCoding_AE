# ChannelCoding_AE
This code is for paper: On the Design of Channel Coding Autoencoders with Arbitrary Rates for ISI Channels 

This paper presents an autoencoder-based channel coding scheme in the presence of inter-symbol interference (ISI) and additive white Gaussian noise (AWGN), supporting arbitrary coding rates. \
Both the transmitter and receiver of the proposed autoencoder utilize bi-directional gated recurrent unit (Bi-GRU) layers. 
ChannelCoding AE outperforms conventional convolutional codes significantly and beats LDPC codes in the low signal-to-noise ratio regime over ISI channels.

Required library: test on Python 3.6.11+ PyTorch 1.0.


## Run experiments:

Before running, run 'mkdir ./logs/' to put folder for logs, run 'mkdir ./tmp/' to put folder for weights.

(1) To train the model (R = 2/3):

    python main.py -encoder rate_high -decoder rate_high -enc_num_unit 25 -enc_num_layer 2 -enc_act linear -dec_num_unit 100 -dec_num_layer 2 -dec_act sigmoid -channel isi -code_rate_k 2 -code_rate_n 3 -train_enc_channel_low 0 -train_enc_channel_high 8 -snr_test_start 0 -snr_test_end 12.0 -snr_points 13 -is_parallel 1 -is_interleave 0 -train_dec_channel_low 0 -train_dec_channel_high 8 -dec_lr 0.001 -enc_lr 0.001 -num_block 100000 -batch_size 1000 -train_channel_mode block_norm -test_channel_mode block_norm --print_test_traj -loss bce -num_epoch 80 -joint_train 1

(2) To test the model (R = 2/3), just enforce `-num_epoch 0`:
    
    python main.py -encoder rate_high -decoder rate_high -enc_num_unit 25 -enc_num_layer 2 -enc_act linear -dec_num_unit 100 -dec_num_layer 2 -dec_act sigmoid -channel isi -code_rate_k 2 -code_rate_n 3 -train_enc_channel_low 0 -train_enc_channel_high 8 -snr_test_start 0 -snr_test_end 12.0 -snr_points 13 -is_parallel 1 -is_interleave 0 -train_dec_channel_low 0 -train_dec_channel_high 8 -dec_lr 0.001 -enc_lr 0.001 -num_block 100000 -batch_size 1000 -train_channel_mode block_norm -test_channel_mode block_norm --print_test_traj -loss bce -num_epoch 0 -joint_train 1 -init_nw_weight ./tmp/torch_model_xxxx.pt (the model you saved at training)

