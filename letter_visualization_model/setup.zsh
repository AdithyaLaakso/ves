#!/bin/zsh

sudo nvidia-smi -caa
# ----------------------------
# CPU / OpenMP / MKL settings
# ----------------------------
export OMP_NUM_THREADS=2                 # Match to number of physical cores (adjust to your CPU)
export OMP_SCHEDULE=STATIC                # Static scheduling for uniform workloads
export OMP_PROC_BIND=CLOSE                # Bind threads close to master for cache locality
export GOMP_CPU_AFFINITY="0-2"          # Pin threads to cores 0-19
export KMP_AFFINITY=granularity=fine,compact,1,0  # Fine granularity, compact placement
export KMP_BLOCKTIME=0                    # Threads sleep immediately when idle (better for GPU-heavy loops)
export CUDA_LAUNCH_BLOCKING=1

# ----------------------------
# PyTorch / CUDA settings
# ----------------------------
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True
export TORCH_DISABLE_TF32_LEGACY_API=1

# ----------------------------
# Optional PyTorch tuning
# ----------------------------
export CUDNN_BENCHMARK=1

# ----------------------------
# Logging
# ----------------------------
export TORCH_TRACE=./logs.txt
# export TORCH_LOGS=
export TORCHDYNAMO_VERBOSE=1
# export TORCH_COMPILE_DEBUG=

mkdir -p checkpoints/
rm -f checkpoints/* &> /dev/null || true

mkdir -p logs_archive/
mv -f logs/* logs_archive/
rm *.stamp

stamp=$(date +%s)
file_name=$stamp".stamp"
touch $file_name
killall tensorboard
nohup tensorboard --logdir ./logs/$stamp &


python3 train_reconstruction.py && python3 visualize_model.py
