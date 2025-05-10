#!/usr/bin/bash
set -ex

NGPU=1  # 单卡训练
export LOG_RANK=0
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_1b.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# 设置 LOCAL_RANK、RANK、WORLD_SIZE、MASTER_ADDR 和 MASTER_PORT 环境变量
export LOCAL_RANK=${LOCAL_RANK:-0}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torchtitan.train --job.config_file ${CONFIG_FILE} $overrides

# #!/usr/bin/bash
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# set -ex

# # use envs as local overrides for convenience
# # e.g.
# # LOG_RANK=0,1 NGPU=4 ./run_train.sh
# NGPU=${NGPU:-"1"}
# export LOG_RANK=${LOG_RANK:-0}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}

# overrides=""
# if [ $# -ne 0 ]; then
#     overrides="$*"
# fi

# TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
# TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
# torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
# --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
# -m torchtitan.train --job.config_file ${CONFIG_FILE} $overrides