# Reproducibility Environment Report

Generated (UTC): `2026-08-21T04:20:13+00:00`  

## Source Revision

- Git metadata: unavailable (run inside the repository checkout).

## Host

- OS: `Windows-11-10.0.26200-SP0`
- Architecture: `AMD64`
- Processor: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel`
- Python: `3.13.9` (CPython)

## PyTorch and Accelerator Runtime

- PyTorch: `2.12.1+cu132`
- CUDA used to build PyTorch: `13.2`
- CUDA available: `True`
- cuDNN: `92000`
- Deterministic algorithms at capture: `False`
- cuDNN deterministic / benchmark: `False` / `False`
- TF32 matmul / cuDNN: `False` / `True`
- GPU 0: `NVIDIA GeForce RTX 3090 Ti`; compute capability `8.6`; `23.99 GiB`
- NVIDIA-SMI GPU 0: `NVIDIA GeForce RTX 3090 Ti`; driver `610.62`; `24564 MiB`

## CUDA and Compiler Toolchain

- nvcc: `N/A`
- Python compiler: `MSC v.1944 64 bit (AMD64)`
- cl: `���� x64 �� Microsoft (R) C/C++ �Ż������� 19.44.35223 ��`

## Core Python Packages

| Distribution | Version | Provenance |
|---|---:|---|
| adafactor8bit | 0.4.3 | package index / unspecified |
| torch | 2.12.1+cu132 | package index / unspecified |
| transformers | 5.13.0 | package index / unspecified |
| datasets | 4.4.1 | package index / unspecified |
| accelerate | 1.14.0 | package index / unspecified |
| bitsandbytes | 0.50.0.dev0 | local wheel: bitsandbytes-1.33.7.preview-py3-none-win_amd64.whl; sha256=8f86ae3b997d27aab835b6ce1e6b749e22ed6fa5755c7a043bb6fe55a06e818d |
| flash_attn | 2.8.3 | local wheel: flash_attn-2.8.3-cp313-cp313-win_amd64.whl; sha256=681b9e23b8e94d51065a83a4e86c110a9e53f480ffff80108fff098400e8a6a9 |
| numpy | 2.4.6 | package index / unspecified |
| tensorboard | 2.20.0 | package index / unspecified |
| tokenizers | 0.22.1 | package index / unspecified |
| safetensors | 0.8.0 | package index / unspecified |
| huggingface_hub | 1.9.0 | package index / unspecified |
| ninja | 1.13.0 | package index / unspecified |
| triton | not installed | -- |

## Selected Environment Variables

- None of the selected CUDA/PyTorch variables were set in the capture process.

## Interpretation Notes

- Local installation paths are intentionally omitted. Local wheels are identified by artifact basename and recorded digest when available.
- Runtime flags above describe the capture process. Experiment scripts remain the source of truth for flags explicitly set during training.
- This curated report records result-relevant dependencies; it is not a complete dump of every transitive package in the environment.
