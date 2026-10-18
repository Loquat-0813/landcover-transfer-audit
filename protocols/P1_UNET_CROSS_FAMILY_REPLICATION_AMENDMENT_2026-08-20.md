# Protocol Amendment: Full-Budget Shuffle Construction

**Parent protocol:** `p1-unet-cross-family-v1`  
**Date:** 2026-08-20  
**Scope:** `unet_b1_shuffle` schedule construction only

## Triggering pre-training failure

The first full-budget schedule attempt for seed 19 was rejected before any
`unet_b1_shuffle` optimizer update, checkpoint, source-validation metric, or
target replay. Its archived gate values were:

```text
native_pair_count              = 0
mask_density_multiset_preserved = true
mean_abs_density_difference     = 0.2054895817
p95_abs_density_difference      = 0.6338369370
max_abs_density_difference      = 1.0
```

The failure exposed an implementation defect: the schedule builder randomized
the order in which it tried cyclic offsets in density-sorted records. A large
offset is a valid derangement but is not a density-matched assignment.

## Amendment

The builder now tries the one-position cyclic shift first, then increasingly
larger offsets only when necessary to resolve repeated source identifiers. It
retains the same-region, no-native-pair, and exact density-multiset conditions.
The numerical eligibility gates are unchanged:

- mean absolute density difference at most `0.005`;
- 95th percentile absolute density difference at most `0.05`.

Before any replacement `unet_b1_shuffle` training, the three full-budget
schedules must be materialized by `audit_unet_full_shuffle_schedule.py`,
independently pass these unchanged gates, and be archived with SHA-256. Each
replacement training run consumes its precomputed seed-matched schedule and
records that input digest.

This amendment does not alter the completed `unet_b0`, `unet_b0_half`, or
`unet_b1` runs. The rejected seed-19 schedule remains part of the execution
ledger and receives no efficacy interpretation.
