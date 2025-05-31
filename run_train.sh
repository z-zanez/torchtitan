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
