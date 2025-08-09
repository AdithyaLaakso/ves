export OMP_NUM_THREADS=10
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="0-9"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
