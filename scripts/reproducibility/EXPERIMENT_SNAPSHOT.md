# Experiment source snapshot

This file separates the optimizer revision used to run the experiments from
the later commit that assembles this artifact repository.

## Verified implementation anchor

- Repository: `https://github.com/yanfeiwong/adafactor-8bit`
- Package version: `0.4.3`
- Experiment code commit: `4f544d8b8eebaf50053a4e8a27096e79b049b480`
- Commit subject: `feat: add conf_quant_type for independent confidence state quantization`
- Verification status: full hash verified by the author on 2026-08-21

This is the implementation lineage used by the analyzed experiments. Later
commits that change README content, links, author-name presentation, or other
documentation do not replace the experimental source identity.

## Artifact freeze fields

These fields distinguish the experiment implementation from the later public
artifact assembly:

- Full experiment commit: `4f544d8b8eebaf50053a4e8a27096e79b049b480`
- Initial artifact assembly commit: `8ed6eede5a1228c7e5347e8f17b96017214a94b6`
- Public release tag: `not assigned`
- Artifact DOI or immutable archive URL: `not assigned`
- Working tree clean at initial archive time: `yes`

The assembly commit should identify the repository containing this file. It
must not be substituted for the experiment implementation commit above.

## Numerical reproducibility boundary

Fixed seeds control stochastic data/model operations where supported. The
experiment scripts do not enable PyTorch's global deterministic-algorithm
mode. Custom factored CUDA kernels accumulate row or column statistics with
`atomicAdd`, so CUDA thread scheduling can change floating-point accumulation
order and produce small backend or run-to-run differences.

The smoke report therefore distinguishes strict reference-algorithm alignment
from CUDA numerical consistency. This is a documented implementation property,
not an exception added for a particular result.
