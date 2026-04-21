# Pilot Evaluation Report for the Revised VES Pipeline

## 1. Purpose

This report records the initial execution and scaling tests performed on the revised VES pipeline. Its purpose is to document what has already been validated locally, compare CPU and GPU behavior across test environments, and clarify what those results imply for the next stage of compute planning.

## 2. Revised Pipeline Context

This pilot evaluates the revised VES training setup, which differs from the earlier workflow in several important ways. The current configuration uses a manifest-based `ALPUB_v2` dataset representation, document-level train/validation splitting, mild sampler-level class rebalancing, metadata-level label normalization, and on-the-fly degradation of clean uncial images for reconstruction training. The present report is therefore not a benchmark of the older pipeline, but of the revised one.

## 3. Local CPU Validation

### 3.1 Objective

The first stage of validation was designed to confirm that the revised pipeline could run successfully in the present local environment after corrections to script robustness, interpreter selection, and smoke-test behavior.

### 3.2 Environment

- host: local workstation
- execution mode: CPU-only
- Python: `/usr/bin/python3`
- PyTorch: `2.8.0+rocm6.4`
- GPU acceleration: unavailable in practice on this machine

### 3.3 Successful Smoke Test

- dataset size: `64`
- epochs: `1`
- batch size: `4`
- result: completed successfully
- final output:
  - `Epoch 1/1 | train_loss=7.1367 val_loss=7.0424`
  - checkpoint saved
  - final model saved

### 3.4 Larger Local CPU Validation Run

- dataset size: `2048`
- epochs: `1`
- batch size: `8`
- result: exceeded hard timeout at `1200` seconds
- interpretation: local CPU execution is valid for smoke-scale testing but becomes impractically slow at moderate subset scale

## 4. External GPU Validation

### 4.1 Objective

This stage is intended to determine whether the revised pipeline becomes practically usable on modern NVIDIA hardware.

### 4.2 Environment

- host: Hugo's system
- GPU: `[to be filled in]`
- Python: `[to be filled in]`
- PyTorch: `[to be filled in]`

### 4.3 Test Configuration

- dataset size: `2048`
- epochs: `1`
- batch size: `16`
- TensorBoard: disabled
- visualization: disabled

### 4.4 Result

- completion status: `[pending]`
- runtime: `[pending]`
- train loss: `[pending]`
- val loss: `[pending]`

## 5. Comparison and Interpretation

This section should compare the local CPU and external GPU results directly. The key question is not whether the model can run at all, but whether it can be run fast enough to support meaningful iteration. If the GPU run completes cleanly and materially faster than the local CPU run, that would strongly support the case for a bounded pilot on research computing infrastructure.

## 6. Implications for Next Compute Step

The local CPU results already suggest that the revised pipeline has moved beyond pure proof-of-execution and into a phase where compute availability meaningfully constrains progress. If the external GPU run confirms a substantial improvement in runtime without destabilizing training, the next logical step is a modest pilot on modern research GPU infrastructure, ideally beginning with a one-GPU subset benchmark and then scaling to a larger one-epoch run.

## 7. Provisional Conclusion

At present, the revised VES pipeline has been validated locally at smoke-test scale and shown to be too slow for efficient moderate-scale iteration on commodity CPU hardware. External GPU validation is in progress. The significance of the present report is therefore provisional but already useful: it establishes that the pipeline is real, runnable, and constrained primarily by compute rather than by complete implementation failure.
