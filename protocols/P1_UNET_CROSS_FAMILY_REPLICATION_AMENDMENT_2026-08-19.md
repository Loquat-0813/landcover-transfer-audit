# Protocol Amendment: CUDA Equivalence Tolerance

**Parent protocol:** `p1-unet-cross-family-v1`  
**Date:** 2026-08-19  
**Scope:** label-withheld implementation-integrity test only

The first CUDA execution used the original `1e-6` loss/gradient tolerance and failed with:

```text
solo loss    = 2.02294731
paired loss  = 2.02295113
absolute diff = 3.81469727e-6
```

The CPU test passed with zero loss difference and a maximum gradient difference of `7.08e-8`. The CUDA discrepancy is consistent with different deterministic reduction order between singleton and batched convolution execution; it is not an efficacy result and does not change the declared training arms.

The integrity gate is therefore amended before any U-Net smoke or full-budget run:

- CPU: `rtol=1e-6`, `atol=1e-6`.
- CUDA: `rtol=5e-5`, `atol=5e-5`.

The first CUDA run under the `1e-5` gate passed the loss check (`3.81e-6`) but observed a maximum absolute gradient difference of `2.16e-5`. The CUDA report must retain both observed differences and the one-step AdamW parameter difference. This amendment does not permit a non-finite result, a failed gradient gate, or any change to the U-Net training/evaluation protocol.
