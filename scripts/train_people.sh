#!/bin/bash

OPTS=""
OPTS+="--id PEOPLE_with_voice_normalized_3000speakers_hopsize_wav "
OPTS+="--list_train data/train.csv "
OPTS+="--list_val data/val.csv "

# Models
OPTS+="--arch_sound unet5 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 13 "
OPTS+="--stride_frames 2 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 48000 "
OPTS+="--audRate 16000 "

# learning params
OPTS+="--num_gpus 2 "
OPTS+="--workers 16 "
OPTS+="--batch_size_per_gpu 10 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-4 "
OPTS+="--lr_synthesizer 1e-4 "
OPTS+="--num_epoch 1000 "
OPTS+="--lr_steps 100 200 "

#OPTS+="--weights_sound '' "
#OPTS+="--weights_frame '' "
#OPTS+="--weights_synthesizer '' "

#Hops and window
#OPTS+="--stft_frame 400 "
#OPTS+="--stft_hop 160"

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

python -W ignore -u mainpeople.py $OPTS
