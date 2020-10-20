#!/bin/bash

OPTS=""
OPTS+="--id LIPS_bilstm_fastdataloader "
OPTS+="--list_train data/trainlrs2s_fusiondebug.csv "
OPTS+="--list_val data/vallrs2s_fusion.csv "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 75 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 25 "
OPTS+="--imgSize 256 "

# audio-related
OPTS+="--audLen 48000 "
OPTS+="--audRate 16000 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 16 "
OPTS+="--batch_size_per_gpu 2 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "

#Hops and window
#OPTS+="--stft_frame 400 "
#OPTS+="--stft_hop 160"

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 5 "

python -W ignore -u mainpeople_fusion_lrs2.py $OPTS
