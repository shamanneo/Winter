# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#export GLOO_SOCKET_IFNAME=eth0
python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 20100 )) \
	--num-gpus 2 \
	--config configs/lip/m2fp_R101_bs16_288k.yaml \
	OUTPUT_DIR ../result/training_dir/lip/work \
