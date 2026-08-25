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

Fill these after the first clean commit of this repository, and update them
again if a release tag or immutable archive is created:

- Full experiment commit: `4f544d8b8eebaf50053a4e8a27096e79b049b480`
- Artifact assembly commit: `TBD after initial artifact commit`
- Public release tag: `TBD`
- Artifact DOI or immutable archive URL: `TBD`
- Working tree clean at archive time: `TBD`

The assembly commit should identify the repository containing this file. It
must not be substituted for the experiment implementation commit above.

## Numerical reproducibility boundary

Fixed seeds and deterministic library settings control stochastic data/model
operations where supported. They do not guarantee bitwise equality for custom
factored CUDA kernels that accumulate row or column statistics with
`atomicAdd`. CUDA thread scheduling can change floating-point accumulation
order, producing small backend or run-to-run differences.

The smoke report therefore distinguishes strict reference-algorithm alignment
from CUDA numerical consistency. This is a documented implementation property,
not an exception added for a particular result.
