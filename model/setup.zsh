#!/bin/zsh

setopt NULL_GLOB

# if ! git diff --quiet || ! git diff --cached --quiet; then
# 	echo "⚠️ Script can only run on a commited state. It is frustrating but think of all the progress you have lost..."
#   exit 1
# fi

commit_hash=$(git rev-parse --short HEAD)

# sudo nvidia-smi -caa
# ----------------------------
# CPU / OpenMP / MKL settings
# ----------------------------
export VES_SMOKE_TEST="${VES_SMOKE_TEST:-1}"
export VES_FORCE_CPU="${VES_FORCE_CPU:-1}"
export VES_SIZE_PROFILE="${VES_SIZE_PROFILE:-96}"
export VES_NUM_EPOCHS="${VES_NUM_EPOCHS:-1}"
export VES_BATCH_SIZE="${VES_BATCH_SIZE:-4}"
export VES_MAX_SIZE="${VES_MAX_SIZE:-64}"
export VES_NUM_WORKERS="${VES_NUM_WORKERS:-0}"
export VES_TORCH_THREADS="${VES_TORCH_THREADS:-2}"
export VES_TORCH_INTEROP_THREADS="${VES_TORCH_INTEROP_THREADS:-1}"
export VES_RUN_TENSORBOARD="${VES_RUN_TENSORBOARD:-0}"
export VES_RUN_VISUALIZE="${VES_RUN_VISUALIZE:-0}"
export VES_WARN_TIMEOUT="${VES_WARN_TIMEOUT:-600}"
export VES_HARD_TIMEOUT="${VES_HARD_TIMEOUT:-1200}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"                 # Match to number of physical cores (adjust to your CPU)
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
# export TORCH_TRACE=./logs.txt
# export TORCH_LOGS=
export TORCHDYNAMO_VERBOSE=1
# export TORCH_COMPILE_DEBUG=
export PYTHONFAULTHANDLER=1

mkdir -p checkpoints/ checkpoints_archive/
checkpoint_files=(checkpoints/*)
if (( ${#checkpoint_files[@]} )); then
	mv -f $checkpoint_files checkpoints_archive/
	rm -rf $checkpoint_files
fi

mkdir -p logs_archive/
log_files=(logs/*)
if (( ${#log_files[@]} )); then
	mv -f $log_files logs_archive/
fi
stamp_files=(*.stamp)
if (( ${#stamp_files[@]} )); then
	rm -f $stamp_files
fi

file_name=$commit_hash".stamp"
touch $file_name

if [ -n "${PYTHON_BIN:-}" ]; then
	python_bin="$PYTHON_BIN"
else
	for candidate in /usr/bin/python3 python3 /usr/bin/python /bin/python3; do
		if ! command -v "$candidate" >/dev/null 2>&1; then
			continue
		fi
		resolved_bin="$(command -v "$candidate")"
		if "$resolved_bin" - <<'PY' >/dev/null 2>&1
import torch
PY
		then
			python_bin="$resolved_bin"
			break
		fi
	done
fi

if [ -z "${python_bin:-}" ]; then
	echo "Unable to find a Python interpreter with torch installed." >&2
	exit 1
fi

"$python_bin" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"Python preflight failed using {sys.executable}: {exc}", file=sys.stderr)
    raise
print(f"Using Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
PY

echo "VES size profile: ${VES_SIZE_PROFILE}"

if [ "$VES_RUN_TENSORBOARD" = "1" ]; then
	killall tensorboard 2>/dev/null || true
	nohup "$python_bin" -m tensorboard.main --logdir ./logs --port=6006 ./logs/$commit_hash &
fi

if command -v timeout >/dev/null 2>&1; then
	timeout --foreground -k 30 "$VES_HARD_TIMEOUT" "$python_bin" train_reconstruction.py &
	train_pid=$!
	(
		sleep "$VES_WARN_TIMEOUT"
		if kill -0 "$train_pid" 2>/dev/null; then
			echo "Smoke test still running after ${VES_WARN_TIMEOUT}s"
		fi
	) &
	warn_pid=$!
	wait "$train_pid"
	train_status=$?
	kill "$warn_pid" 2>/dev/null || true
	if [ "$train_status" -eq 124 ]; then
		echo "Smoke test hit hard timeout after ${VES_HARD_TIMEOUT}s"
	fi
	if [ "$train_status" -ne 0 ]; then
		exit "$train_status"
	fi
else
	"$python_bin" train_reconstruction.py
fi

if [ "$VES_RUN_VISUALIZE" = "1" ]; then
	"$python_bin" visualize_model.py
fi
