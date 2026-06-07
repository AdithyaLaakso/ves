AWS GPU Validation Report for Hugo

Date: 2026-06-07

Purpose

Summarize the first AWS GPU validation runs so Hugo can compare them with his
earlier training attempts and help identify whether his plateau was caused by
model behavior, environment setup, data availability, or invocation details.

Important caveat

These results are not an apples-to-apples comparison with Hugo's earlier runs.
The code path, model setup, runtime environment, and possibly dependency
versions have changed. The useful comparison is therefore diagnostic rather than
competitive: this report shows that the current branch can train on GPU, can
load the full ALPUB manifest image set, and does not show an early plateau on
bounded subset runs.

Environment

- Repository branch: `uncial-pilot-prep`
- Commit: `6e1be9724`
- EC2 AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11, Ubuntu 24.04
- Python: `/opt/pytorch/bin/python`
- PyTorch: `2.11.0+cu130`
- CUDA build reported by PyTorch: `13.0`
- CUDA available to PyTorch: `True`
- GPU: NVIDIA L4
- Dataset path on EC2: `/home/ubuntu/ves/ALPUB_v2/images`
- Manifest preflight: `205797` total records, `205797` loadable records,
  `0` missing records, `24` classes

Warnings observed

The AMI emits this warning when importing torch:

```text
FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.
```

This warning did not block training. CUDA was available and the runs trained on
the NVIDIA L4 GPU.

Run summary

| Run | Samples | Epochs | Batch | Size profile | Final train loss | Final validation loss | Run directory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Smoke | 64 | 1 | 2 | 64 | 7.3422 | 7.2459 | `model/runs/20260607T153751Z-6e1be9724` |
| Moderate | 2048 | 1 | 8 | 96 | 6.6553 | 6.4453 | `model/runs/20260607T155143Z-6e1be9724` |
| Moderate curve | 2048 | 5 | 8 | 96 | 6.3772 | 6.2820 | `model/runs/20260607T155541Z-6e1be9724` |
| Larger subset | 8192 | 5 | 8 | 96 | 6.1312 | 5.5344 | `model/runs/20260607T155903Z-6e1be9724` |
| Larger subset, interrupted | 32768 | 4 confirmed of 5 | 8 | 96 | 4.2209 | 4.0132 | `model/runs/20260607T160908Z-6e1be9724` |

Moderate curve: 2048 samples, 5 epochs

```text
Epoch 1/5 | train_loss=6.6568 val_loss=6.4450
Epoch 2/5 | train_loss=6.4891 val_loss=6.3806
Epoch 3/5 | train_loss=6.4165 val_loss=6.3375
Epoch 4/5 | train_loss=6.4323 val_loss=6.3103
Epoch 5/5 | train_loss=6.3772 val_loss=6.2820
```

Larger subset: 8192 samples, 5 epochs

```text
Epoch 1/5 | train_loss=6.5044 val_loss=6.2937
Epoch 2/5 | train_loss=6.3726 val_loss=6.2574
Epoch 3/5 | train_loss=6.3355 val_loss=6.2277
Epoch 4/5 | train_loss=6.3057 val_loss=6.1932
Epoch 5/5 | train_loss=6.1312 val_loss=5.5344
```

Larger subset: 32768 samples, interrupted during epoch 5

```text
Epoch 1/5 | train_loss=6.3739 val_loss=6.2416
Epoch 2/5 | train_loss=5.3969 val_loss=4.4343
Epoch 3/5 | train_loss=4.4506 val_loss=4.1182
Epoch 4/5 | train_loss=4.2209 val_loss=4.0132
```

The run reached `On step 5` and reported CUDA memory allocation for epoch 5,
but the SSH session later disconnected with `client_loop: send disconnect:
Broken pipe`. The EC2 instance then showed `2/3` status checks passed and was
stopped manually to avoid continued compute charges. Treat this run as
confirmed through epoch 4 only unless later disk inspection shows an epoch 5
checkpoint or final `new.pth` from this run.

Artifact verification

For the 2048-sample 5-epoch run, EC2 contained all five checkpoints and the
final model:

```text
runs/20260607T155541Z-6e1be9724/checkpoints/0-1.pth
runs/20260607T155541Z-6e1be9724/checkpoints/0-2.pth
runs/20260607T155541Z-6e1be9724/checkpoints/0-3.pth
runs/20260607T155541Z-6e1be9724/checkpoints/0-4.pth
runs/20260607T155541Z-6e1be9724/checkpoints/0-5.pth
runs/20260607T155541Z-6e1be9724/new.pth
```

For the 8192-sample 5-epoch run, EC2 also contained all five checkpoints and
the final model:

```text
runs/20260607T155903Z-6e1be9724/checkpoints/0-1.pth
runs/20260607T155903Z-6e1be9724/checkpoints/0-2.pth
runs/20260607T155903Z-6e1be9724/checkpoints/0-3.pth
runs/20260607T155903Z-6e1be9724/checkpoints/0-4.pth
runs/20260607T155903Z-6e1be9724/checkpoints/0-5.pth
runs/20260607T155903Z-6e1be9724/new.pth
```

Interpretation

The absolute loss values should not be read as percentages or accuracy. In the
current settings, the segmentation loss is a weighted custom objective, mainly
`10 * focal_loss` plus a reconstruction/SSIM-style term. Lower is better, but
the numeric value is meaningful primarily as a trend across comparable runs.

The bounded runs do not show the early plateau that Hugo reported. Validation
loss improved steadily on the confirmed multi-epoch runs:

- 2048 samples: validation loss improved from `6.4450` to `6.2820`.
- 8192 samples: validation loss improved from `6.2937` to `5.5344`.
- 32768 samples: validation loss improved from `6.2416` to `4.0132` by the
  last confirmed epoch before interruption.

There is also no obvious overfitting signal in these short runs. A typical
overfitting pattern would be training loss decreasing while validation loss
increases. Here, validation loss continued to improve.

The current evidence suggests that the model is still undertrained rather than
plateaued. The 8192-sample run improved materially at epoch 5, and the
32768-sample run improved sharply by epoch 4. This supports either rerunning the
32768-sample test under a more robust session manager or proceeding to a larger
run only after checkpoint-resume behavior is clarified.

Questions for Hugo

To compare this with the earlier plateau, the most useful details from Hugo's
side would be:

- exact commit or model version used
- GPU model and CUDA/PyTorch versions
- whether `torch.cuda.is_available()` returned true
- dataset size and whether a manifest preflight or equivalent file check passed
- batch size, epoch count, learning rate, and image size/profile
- loss function and loss weights used
- per-epoch training and validation loss curves
- whether the plateau occurred in both training and validation loss, or only
  validation loss
- any error messages, CUDA warnings, CPU fallback behavior, or out-of-memory
  symptoms

Suggested next step

Before a full dataset run, restart the instance, inspect the interrupted run
directory, and use a session manager such as `tmux` for any long training job.
The first command after reconnecting should be:

```bash
cd /home/ubuntu/ves/model
ls -lh runs/20260607T160908Z-6e1be9724/checkpoints
ls -lh runs/20260607T160908Z-6e1be9724/new.pth 2>/dev/null || true
```

Resume from the best confirmed checkpoint inside `tmux`:

```bash
cd /home/ubuntu/ves/model

PYTHON_BIN=/opt/pytorch/bin/python \
VES_RESUME_FROM=runs/20260607T195744Z-6e1be9724/checkpoints/0-4.pth \
VES_SMOKE_TEST=0 \
VES_FORCE_CPU=0 \
VES_MAX_SIZE=32768 \
VES_BATCH_SIZE=8 \
VES_NUM_EPOCHS=1 \
VES_SIZE_PROFILE=96 \
VES_SEED=42 \
./setup.zsh
```

This uses the epoch 4 weights from the interrupted 32768-sample run and trains
one additional epoch. If validation loss continues to trend down and the run
finishes cleanly, a full dataset run is justified.
