# Adafactor8Bit Smoke Test Report

**Started**: 2026-08-20T21:57:58+08:00
**Finished**: 2026-08-20T21:58:09+08:00
**Selected part**: `all`
**Device**: `cuda`
**CUDA device**: `NVIDIA GeForce RTX 3090 Ti`
**CUDA kernel**: Available

## Summary

- ✅ Passed: 171
- ❌ Failed: 0
- ⚠️ Warnings: 4
- ⏭️ Skipped: 0

## Configuration

- Seed: `42`
- ALIGN_DIM: `256`
- MATRIX_DIM: `2048`
- Reference alignment tolerance: `1e-05` (strict)
- CUDA/backend consistency PASS threshold: `< 1e-05`
- CUDA/backend consistency FAIL threshold: `>= 5e-03`

## Result Policies

- **Reference Algorithm Alignment**: strict semantic/reference check; no warning band.
- **CUDA Numerical Consistency**: permits the documented warning band for CUDA accumulation-order and backend numerical effects.
- **Functional / Regression**: behavioral, state, checkpoint, memory, and API regression checks.

## Detailed Results

| Status | Policy | Test Name | Details |
|---|---|---|---|
| ✅ PASS | Reference Algorithm Alignment | [py] Adafactor vs HF | diff=3.73e-09 worst=weight tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [py] Adafactor mixed | diff=2.38e-07 worst=1.weight tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [py] AdamW vs torch | diff=2.24e-08 tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [py] RMSprop vs torch | diff=0.00e+00 tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [py] CAME vs official (2D) | diff=4.77e-07 worst=weight tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [py] CAME mixed model (2D + 1D) | diff=3.26e-06 worst=2.weight tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [py] APOLLO vs official | diff=7.45e-09 worst=weight tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [cuda] Adafactor vs HF | diff=7.45e-09 worst=weight tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [cuda] Adafactor mixed | diff=1.19e-07 worst=1.weight tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [cuda] AdamW vs torch | diff=1.12e-07 tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [cuda] RMSprop vs torch | diff=4.47e-08 tol=1e-05 |
| ✅ PASS | Reference Algorithm Alignment | [cuda] CAME vs official (2D) | diff=1.43e-06 worst=weight tol=1e-05 |
| ⚠️ WARN | CUDA Numerical Consistency | [cuda] CAME mixed model (2D + 1D) | Δp=1.24e-05 worst=2.weight pass<1e-05 fail>=5e-03 expected accumulation-order variation from factored CUDA atomicAdd reductions |
| ✅ PASS | Reference Algorithm Alignment | [cuda] APOLLO vs official | diff=7.45e-09 worst=weight tol=1e-05 |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py adafactor al8 | Δp=1.12e-08 worst=weight pass<1e-05 fail>=5e-03 |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py adamw al8+uf8 | Δp=5.96e-08 worst=weight pass<1e-05 fail>=5e-03 |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py adamw al8+d8 | Δp=7.45e-08 worst=weight pass<1e-05 fail>=5e-03 |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py adamw al16+uf8 | Δp=6.71e-08 worst=weight pass<1e-05 fail>=5e-03 |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py adamw vfp32+uf8 | Δp=0.00e+00 pass<1e-05 fail>=5e-03 |
| ⚠️ WARN | CUDA Numerical Consistency | cuda≈py came al8+uf8 | Δp=1.25e-04 worst=weight pass<1e-05 fail>=5e-03 |
| ⚠️ WARN | CUDA Numerical Consistency | cuda≈py came al8+d8 | Δp=1.28e-04 worst=weight pass<1e-05 fail>=5e-03 |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py came al8+uf8+cal16 | Δp=4.89e-06 worst=weight pass<1e-05 fail>=5e-03 |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py came al8+uf8+cfp32 | Δp=1.67e-06 worst=weight pass<1e-05 fail>=5e-03 |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py apollo M=uf8 | Δp=7.45e-09 pass<1e-05 fail>=5e-03 projection seed-dependent |
| ✅ PASS | CUDA Numerical Consistency | cuda≈py apollo M=d8 | Δp=7.45e-09 pass<1e-05 fail>=5e-03 projection seed-dependent |
| ✅ PASS | Functional / Regression | [py] adafactor V=al8 M=noM | loss 104.05→-364.07 mem=4KB |
| ✅ PASS | Functional / Regression | [py] adafactor V=al16 M=noM | loss 104.05→-364.08 mem=8KB |
| ✅ PASS | Functional / Regression | [py] adafactor V=vfp32 M=noM | loss 104.05→-364.08 mem=16KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al8 M=uf4 | loss 104.05→-36193.97 mem=6288KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al8 M=uf8 | loss 104.05→-36557.30 mem=8272KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al8 M=d4 | loss 104.05→-35888.91 mem=6288KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al8 M=d8 | loss 104.05→-36569.12 mem=8272KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al8 M=mfp32 | loss 104.05→-36599.19 mem=20496KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al16 M=uf4 | loss 104.05→-36195.30 mem=10384KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al16 M=uf8 | loss 104.05→-36558.31 mem=12368KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al16 M=d4 | loss 104.05→-35889.20 mem=10384KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al16 M=d8 | loss 104.05→-36570.07 mem=12368KB |
| ✅ PASS | Functional / Regression | [py] adamw V=al16 M=mfp32 | loss 104.05→-36600.18 mem=24592KB |
| ✅ PASS | Functional / Regression | [py] adamw V=vfp32 M=uf4 | loss 104.05→-36195.30 mem=18560KB |
| ✅ PASS | Functional / Regression | [py] adamw V=vfp32 M=uf8 | loss 104.05→-36558.30 mem=20544KB |
| ✅ PASS | Functional / Regression | [py] adamw V=vfp32 M=d4 | loss 104.05→-35889.20 mem=18560KB |
| ✅ PASS | Functional / Regression | [py] adamw V=vfp32 M=d8 | loss 104.05→-36570.07 mem=20544KB |
| ✅ PASS | Functional / Regression | [py] adamw V=vfp32 M=mfp32 | loss 104.05→-36600.17 mem=32768KB |
| ✅ PASS | Functional / Regression | [py] came V=al8 M=uf8 | loss 104.05→-623639.88 mem=4168KB |
| ✅ PASS | Functional / Regression | [py] came V=al8 M=d8 | loss 104.05→-624001.50 mem=4168KB |
| ✅ PASS | Functional / Regression | [py] came V=al8 M=mfp32 | loss 104.05→-624333.00 mem=16392KB |
| ✅ PASS | Functional / Regression | [py] came V=al16 M=uf8 | loss 104.05→-622857.88 mem=4176KB |
| ✅ PASS | Functional / Regression | [py] came V=al16 M=d8 | loss 104.05→-623025.88 mem=4176KB |
| ✅ PASS | Functional / Regression | [py] came V=al16 M=mfp32 | loss 104.05→-624335.44 mem=16400KB |
| ✅ PASS | Functional / Regression | [py] came V=vfp32 M=uf8 | loss 104.05→-622852.00 mem=4192KB |
| ✅ PASS | Functional / Regression | [py] came V=vfp32 M=d8 | loss 104.05→-623020.00 mem=4192KB |
| ✅ PASS | Functional / Regression | [py] came V=vfp32 M=mfp32 | loss 104.05→-624335.38 mem=16416KB |
| ✅ PASS | Functional / Regression | [py] apollo V=al8 M=uf8 | loss 104.05→-933.23 mem=4176KB |
| ✅ PASS | Functional / Regression | [py] apollo V=al8 M=d8 | loss 104.05→-933.23 mem=4176KB |
| ✅ PASS | Functional / Regression | [py] apollo V=al16 M=uf8 | loss 104.05→-933.23 mem=4184KB |
| ✅ PASS | Functional / Regression | [py] apollo V=al16 M=d8 | loss 104.05→-933.23 mem=4184KB |
| ✅ PASS | Functional / Regression | [py] apollo V=vfp32 M=uf8 | loss 104.05→-933.23 mem=4200KB |
| ✅ PASS | Functional / Regression | [py] apollo V=vfp32 M=d8 | loss 104.05→-933.23 mem=4200KB |
| ✅ PASS | Functional / Regression | [py] came C=cal8 | loss 104.05→-623639.88 mem=4168KB |
| ✅ PASS | Functional / Regression | [py] came C=cal16 | loss 104.05→-623639.88 mem=4172KB |
| ✅ PASS | Functional / Regression | [py] came C=cfp32 | loss 104.05→-623639.88 mem=4180KB |
| ✅ PASS | Functional / Regression | [cuda] adafactor V=al8 M=noM | loss 104.05→-364.07 mem=4KB |
| ✅ PASS | Functional / Regression | [cuda] adafactor V=al16 M=noM | loss 104.05→-364.08 mem=8KB |
| ✅ PASS | Functional / Regression | [cuda] adafactor V=vfp32 M=noM | loss 104.05→-364.08 mem=16KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al8 M=uf4 | loss 104.05→-36193.98 mem=6288KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al8 M=uf8 | loss 104.05→-36557.30 mem=8272KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al8 M=d4 | loss 104.05→-35888.20 mem=6288KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al8 M=d8 | loss 104.05→-36569.12 mem=8272KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al8 M=mfp32 | loss 104.05→-36599.18 mem=20496KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al16 M=uf4 | loss 104.05→-36195.31 mem=10384KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al16 M=uf8 | loss 104.05→-36558.32 mem=12368KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al16 M=d4 | loss 104.05→-35888.48 mem=10384KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al16 M=d8 | loss 104.05→-36570.09 mem=12368KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=al16 M=mfp32 | loss 104.05→-36600.18 mem=24592KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=vfp32 M=uf4 | loss 104.05→-36195.30 mem=18560KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=vfp32 M=uf8 | loss 104.05→-36558.30 mem=20544KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=vfp32 M=d4 | loss 104.05→-35889.20 mem=18560KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=vfp32 M=d8 | loss 104.05→-36570.07 mem=20544KB |
| ✅ PASS | Functional / Regression | [cuda] adamw V=vfp32 M=mfp32 | loss 104.05→-36600.18 mem=32768KB |
| ✅ PASS | Functional / Regression | [cuda] came V=al8 M=uf8 | loss 104.05→-623642.62 mem=4168KB |
| ✅ PASS | Functional / Regression | [cuda] came V=al8 M=d8 | loss 104.05→-624003.94 mem=4168KB |
| ✅ PASS | Functional / Regression | [cuda] came V=al8 M=mfp32 | loss 104.05→-624332.88 mem=16392KB |
| ✅ PASS | Functional / Regression | [cuda] came V=al16 M=uf8 | loss 104.05→-622862.00 mem=4176KB |
| ✅ PASS | Functional / Regression | [cuda] came V=al16 M=d8 | loss 104.05→-623030.50 mem=4176KB |
| ✅ PASS | Functional / Regression | [cuda] came V=al16 M=mfp32 | loss 104.05→-624335.44 mem=16400KB |
| ✅ PASS | Functional / Regression | [cuda] came V=vfp32 M=uf8 | loss 104.05→-622852.00 mem=4192KB |
| ✅ PASS | Functional / Regression | [cuda] came V=vfp32 M=d8 | loss 104.05→-623020.00 mem=4192KB |
| ✅ PASS | Functional / Regression | [cuda] came V=vfp32 M=mfp32 | loss 104.05→-624329.12 mem=16416KB |
| ✅ PASS | Functional / Regression | [cuda] apollo V=al8 M=uf8 | loss 104.05→-933.23 mem=4176KB |
| ✅ PASS | Functional / Regression | [cuda] apollo V=al8 M=d8 | loss 104.05→-933.23 mem=4176KB |
| ✅ PASS | Functional / Regression | [cuda] apollo V=al16 M=uf8 | loss 104.05→-933.23 mem=4184KB |
| ✅ PASS | Functional / Regression | [cuda] apollo V=al16 M=d8 | loss 104.05→-933.23 mem=4184KB |
| ✅ PASS | Functional / Regression | [cuda] apollo V=vfp32 M=uf8 | loss 104.05→-933.23 mem=4200KB |
| ✅ PASS | Functional / Regression | [cuda] apollo V=vfp32 M=d8 | loss 104.05→-933.23 mem=4200KB |
| ✅ PASS | Functional / Regression | [cuda] came C=cal8 | loss 104.05→-623642.62 mem=4168KB |
| ✅ PASS | Functional / Regression | [cuda] came C=cal16 | loss 104.05→-623642.62 mem=4172KB |
| ✅ PASS | Functional / Regression | [cuda] came C=cfp32 | loss 104.05→-623642.75 mem=4180KB |
| ✅ PASS | Functional / Regression | V ordering: al8 < al16 < fp32 | 8272KB < 12368KB < 20544KB |
| ✅ PASS | Functional / Regression | M ordering: uf4 < uf8 < fp32 | 6288KB < 8272KB < 20496KB |
| ✅ PASS | Functional / Regression | C ordering: cal8 < cal16 < cfp32 | 4168KB < 4172KB < 4180KB |
| ✅ PASS | Functional / Regression | Overall: quantized < 60% fp32 | 8272KB vs 32768KB (25.2%) |
| ✅ PASS | Functional / Regression | APOLLO rank: r4 < r64 | 4176KB < 4418KB |
| ✅ PASS | Functional / Regression | State stability (10 extra steps) | 8272KB → 8272KB |
| ✅ PASS | Functional / Regression | [py] block_size > numel |  |
| ✅ PASS | Functional / Regression | [py] frozen params |  |
| ✅ PASS | Functional / Regression | [py] grad accumulation |  |
| ✅ PASS | Functional / Regression | [py] NaN grad recovery |  |
| ✅ PASS | Functional / Regression | [py] wd decoupled |  |
| ✅ PASS | Functional / Regression | [py] wd scaled |  |
| ✅ PASS | Functional / Regression | [py] bf16 model |  |
| ✅ PASS | Functional / Regression | [py] embedding routing |  |
| ✅ PASS | Functional / Regression | [cuda] block_size > numel |  |
| ✅ PASS | Functional / Regression | [cuda] frozen params |  |
| ✅ PASS | Functional / Regression | [cuda] grad accumulation |  |
| ✅ PASS | Functional / Regression | [cuda] NaN grad recovery |  |
| ✅ PASS | Functional / Regression | [cuda] wd decoupled |  |
| ✅ PASS | Functional / Regression | [cuda] wd scaled |  |
| ✅ PASS | Functional / Regression | [cuda] bf16 model |  |
| ✅ PASS | Functional / Regression | [cuda] embedding routing |  |
| ✅ PASS | Functional / Regression | [py] Adafactor V=al8 | fp32=-115.3201 quant=-115.3635 rel=0.0004 |
| ✅ PASS | Functional / Regression | [py] AdamW V=al8 M=uf8 | fp32=-4333.0723 quant=-4333.8887 rel=0.0002 |
| ✅ PASS | Functional / Regression | [py] AdamW V=al8 M=d8 | fp32=-4333.0723 quant=-4218.8877 rel=0.0264 |
| ✅ PASS | Functional / Regression | [py] AdamW V=al8 M=uf4 | fp32=-4333.0723 quant=-4074.7134 rel=0.0596 |
| ✅ PASS | Functional / Regression | [py] AdamW V=al8 M=d4 | fp32=-4333.0723 quant=-4185.8350 rel=0.0340 |
| ✅ PASS | Functional / Regression | [py] AdamW V=al16 M=uf8 | fp32=-4333.0723 quant=-4331.2344 rel=0.0004 |
| ✅ PASS | Functional / Regression | [py] AdamW V=fp32 M=uf8 | fp32=-4333.0723 quant=-4331.2402 rel=0.0004 |
| ✅ PASS | Functional / Regression | [py] AdamW V=al8 M=fp32 | fp32=-4333.0723 quant=-4335.7158 rel=0.0006 |
| ✅ PASS | Functional / Regression | [py] CAME V=al8 M=uf8 | fp32=-157482.4062 quant=-158400.4688 rel=0.0058 |
| ✅ PASS | Functional / Regression | [py] CAME V=al8 M=d8 | fp32=-157482.4062 quant=-158652.5000 rel=0.0074 |
| ✅ PASS | Functional / Regression | [py] CAME V=al8 M=uf8 C=al16 | fp32=-157482.4062 quant=-158402.0312 rel=0.0058 |
| ✅ PASS | Functional / Regression | [py] CAME V=al8 M=uf8 C=fp32 | fp32=-157482.4062 quant=-158402.0000 rel=0.0058 |
| ✅ PASS | Functional / Regression | [cuda] Adafactor V=al8 | fp32=-115.3201 quant=-115.3635 rel=0.0004 |
| ✅ PASS | Functional / Regression | [cuda] AdamW V=al8 M=uf8 | fp32=-4333.0747 quant=-4333.8872 rel=0.0002 |
| ✅ PASS | Functional / Regression | [cuda] AdamW V=al8 M=d8 | fp32=-4333.0747 quant=-4218.8857 rel=0.0264 |
| ✅ PASS | Functional / Regression | [cuda] AdamW V=al8 M=uf4 | fp32=-4333.0747 quant=-4074.7114 rel=0.0596 |
| ✅ PASS | Functional / Regression | [cuda] AdamW V=al8 M=d4 | fp32=-4333.0747 quant=-4185.8335 rel=0.0340 |
| ✅ PASS | Functional / Regression | [cuda] AdamW V=al16 M=uf8 | fp32=-4333.0747 quant=-4331.2324 rel=0.0004 |
| ✅ PASS | Functional / Regression | [cuda] AdamW V=fp32 M=uf8 | fp32=-4333.0747 quant=-4331.2402 rel=0.0004 |
| ✅ PASS | Functional / Regression | [cuda] AdamW V=al8 M=fp32 | fp32=-4333.0747 quant=-4335.7129 rel=0.0006 |
| ✅ PASS | Functional / Regression | [cuda] CAME V=al8 M=uf8 | fp32=-157482.4688 quant=-158399.9375 rel=0.0058 |
| ✅ PASS | Functional / Regression | [cuda] CAME V=al8 M=d8 | fp32=-157482.4688 quant=-158652.2812 rel=0.0074 |
| ✅ PASS | Functional / Regression | [cuda] CAME V=al8 M=uf8 C=al16 | fp32=-157482.4688 quant=-158401.7188 rel=0.0058 |
| ✅ PASS | Functional / Regression | [cuda] CAME V=al8 M=uf8 C=fp32 | fp32=-157482.4688 quant=-158402.0312 rel=0.0058 |
| ✅ PASS | Functional / Regression | [py] roundtrip | max_loss_diff=0.00e+00 |
| ✅ PASS | Functional / Regression | [py] disk roundtrip (al16) | max_loss_diff=0.00e+00 vqt_preserved=True |
| ✅ PASS | Functional / Regression | [py] resume trajectory | max_loss_diff=0.00e+00 |
| ✅ PASS | Functional / Regression | [py] m_quant_type preserved |  |
| ✅ PASS | Functional / Regression | [py] v_quant_type al16 preserved |  |
| ✅ PASS | Functional / Regression | [py] v_quant_type fp32 preserved |  |
| ✅ PASS | Functional / Regression | [py] c_quant_type al16 preserved |  |
| ✅ PASS | Functional / Regression | [py] APOLLO seed preserved | 2 → 2 |
| ✅ PASS | Functional / Regression | [py] empty state save/load |  |
| ✅ PASS | Functional / Regression | [cuda] roundtrip | max_loss_diff=0.00e+00 |
| ✅ PASS | Functional / Regression | [cuda] disk roundtrip (al16) | max_loss_diff=0.00e+00 vqt_preserved=True |
| ✅ PASS | Functional / Regression | [cuda] resume trajectory | max_loss_diff=0.00e+00 |
| ✅ PASS | Functional / Regression | [cuda] m_quant_type preserved |  |
| ✅ PASS | Functional / Regression | [cuda] v_quant_type al16 preserved |  |
| ✅ PASS | Functional / Regression | [cuda] v_quant_type fp32 preserved |  |
| ✅ PASS | Functional / Regression | [cuda] c_quant_type al16 preserved |  |
| ✅ PASS | Functional / Regression | [cuda] APOLLO seed preserved | 2 → 2 |
| ✅ PASS | Functional / Regression | [cuda] empty state save/load |  |
| ✅ PASS | Functional / Regression | [py] determinism adamw | Δloss=0.00e+00 Δ\|p\|=0.00e+00 |
| ✅ PASS | Functional / Regression | [py] determinism adafactor | Δloss=0.00e+00 Δ\|p\|=0.00e+00 |
| ✅ PASS | Functional / Regression | [py] determinism rmsprop | Δloss=0.00e+00 Δ\|p\|=0.00e+00 |
| ✅ PASS | Functional / Regression | [cuda] determinism adamw | Δloss=0.00e+00 Δ\|p\|=0.00e+00 |
| ⚠️ WARN | Functional / Regression | [cuda] determinism adafactor | Δloss=7.63e-06 Δ\|p\|=0.00e+00 (expected: atomicAdd) |
| ✅ PASS | Functional / Regression | [cuda] determinism rmsprop | Δloss=0.00e+00 Δ\|p\|=0.00e+00 |
| ✅ PASS | Functional / Regression | [py] quantize toggle |  |
| ✅ PASS | Functional / Regression | [py] M hot-swap (uf8→d8) |  |
| ✅ PASS | Functional / Regression | [py] V hot-swap (al8→al16) |  |
| ✅ PASS | Functional / Regression | [py] C hot-swap (al8→al16→fp32) |  |
| ✅ PASS | Functional / Regression | [cuda] quantize toggle |  |
| ✅ PASS | Functional / Regression | [cuda] M hot-swap (uf8→d8) |  |
| ✅ PASS | Functional / Regression | [cuda] V hot-swap (al8→al16) |  |
| ✅ PASS | Functional / Regression | [cuda] C hot-swap (al8→al16→fp32) |  |
