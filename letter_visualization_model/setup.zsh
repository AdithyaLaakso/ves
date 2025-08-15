# ----------------------------
# CPU / OpenMP / MKL settings
# ----------------------------
export OMP_NUM_THREADS=1                 # Match to number of physical cores (adjust to your CPU)
export OMP_SCHEDULE=STATIC                # Static scheduling for uniform workloads
export OMP_PROC_BIND=CLOSE                # Bind threads close to master for cache locality
export GOMP_CPU_AFFINITY="0-19"          # Pin threads to cores 0-19
export KMP_AFFINITY=granularity=fine,compact,1,0  # Fine granularity, compact placement
export KMP_BLOCKTIME=0                    # Threads sleep immediately when idle (better for GPU-heavy loops)

# ----------------------------
# PyTorch / CUDA settings
# ----------------------------
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True

# ----------------------------
# Optional PyTorch tuning
# ----------------------------
export CUDNN_BENCHMARK=1

nvidia-smi -caa

python3 train_reconstruction.py && python3 visualize_model.py
