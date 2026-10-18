# P1 U-Net Cross-Family Replication Protocol

**Status:** frozen before implementation, smoke training, or target evaluation  
**Date:** 2026-08-17  
**Protocol ID:** `p1-unet-cross-family-v1`  
**Primary paper:** *Before Attributing Near-Zero IoU to Domain Shift: Auditing Label Coverage in Cross-Dataset Land-Cover Segmentation*

## 1. Purpose and claim boundary

This protocol tests whether the primary LoveDA--FLAIR source-mixture pattern is observed with a standard closed-set segmentation network in addition to the frozen CLIP probe. It does **not** test a label-only causal effect and does not require the U-Net to reproduce the CLIP effect magnitude or a zero B0 rangeland score.

The primary claim eligible after this protocol is limited to: under the declared datasets, mapping, budget, and U-Net implementation, the LoveDA--FLAIR source mixture has a directionally similar or dissimilar effect on the OpenEarthMap rangeland leaf. Any result remains separate from a label-only intervention claim.

## 2. Frozen inputs

| Input | Frozen value |
|---|---|
| Common taxonomy | `ontology/land_cover_taxonomy_v0.yaml`, SHA-256 `a1b2f63e37319518e8c5f04dd408e0954fd5941c7804feeb200dee1092925e54` |
| LoveDA mapping | `ontology/dataset_label_mappings_v0.yaml`, SHA-256 `cd92e8cb472bd74125b235d5e5f0ff5926613c323183f133d2c285e3debec12d` |
| FLAIR mapping | `ontology/dataset_label_mappings_flair_b1_v1.yaml`, SHA-256 `a817fae62d7c5bcc7c8d5eb7001986420c2f2e32dfc337d8097f6be95c723c37` |
| FLAIR raw audit | `audits/flair_b1_full_raw_audit_2026-08-03.json`, SHA-256 `66b1941ef8641648d4dd44d0a05228332012469462805b9e18aa364d5b3fc8be` |
| FLAIR readiness audit | Deterministically derive `audits/flair_b1_source_ready_20260817.json` from the frozen raw audit and the frozen OpenEarthMap release before any smoke or full run; archive its SHA-256 in every U-Net run manifest. |
| FLAIR auxiliary relation | Raw ID 10 only, declared herbaceous vegetation; all other FLAIR raw labels ignored |
| Source sampling rule | `p1train/fixed_budget.py`, SHA-256 `f25e84861dadfa9c72bdb759314fcc88e4455e1a11cfa21b1092bbc6357ade8a` |
| Partial-label loss definition | `p1model/losses.py`, SHA-256 `2b31b40e93e501dbb536e9500f5fabe2b427ba58154b37248b702b81f07330da` |
| Primary target population | Frozen Paris-excluded OpenEarthMap manifest: 2,638 paired rasters in 74 cities, identifier hash `5458f46b24988726842cc7dfd59c4242e237f40b7d90c5ffdceefa30223fc964` |
| Seeds | 19, 37, 73 |
| Input and evaluation tiles | RGB, 512 by 512 pixels; no test-time augmentation or post-processing |
| Optimization budget | 25,220 optimizer updates per run |

The U-Net implementation, package versions, ImageNet weight identifier and downloaded-weight SHA-256 must be recorded in its run manifest before the first full-budget run. They may not change after the first full-budget run without a dated protocol amendment and a new run family.

## 3. Model and optimization

The model family is a standard closed-set U-Net with a ResNet-34 encoder initialized from ImageNet-1K pretrained weights. The decoder and seven-class output head are randomly initialized; the entire network, including the encoder, is optimized. The output channels are ordered as the frozen seven-leaf taxonomy.

The loss is the same taxonomy partial-label negative log-likelihood used by the CLIP probe. Exact labels reduce to ordinary seven-class cross entropy; coarse labels retain the declared descendant-leaf likelihood; ignored pixels make no loss contribution. In the B0 and B0-half arms, herbaceous vegetation has no direct positive source label. It can nevertheless receive indirect negative gradients through the softmax denominator, so a near-zero B0 rangeland score is an empirical outcome, not a protocol requirement.

All runs use AdamW with learning rate `1e-3`, weight decay `1e-4`, and deterministic seed initialization. Each update uses the same source-crop positions as `loveda_positions_for_update`. Batch size is fixed at two samples for B0, B1, and B1-shuffle, and one sample for B0-half, matching the original source-exposure design. The loss reduction is over valid pixels, not batch elements. Any mixed precision, deterministic-kernel setting, worker count, package version, and GPU identifier must be recorded before full training and held fixed across arms where hardware permits.

## 4. Arms

| Arm | Samples per update | LoveDA exposures | Role |
|---|---|---:|---|
| `unet_b0` | two LoveDA crops | 50,440 | base-source baseline |
| `unet_b0_half` | one LoveDA crop | 25,220 | source-exposure control |
| `unet_b1` | one LoveDA crop plus one native FLAIR raw-ID-10 crop | 25,220 | primary source-mixture arm |
| `unet_b1_shuffle` | one LoveDA crop plus one FLAIR RGB crop with a deranged raw-ID-10 mask | 25,220 | native-correspondence arm |

The four arms are each run for all three frozen seeds. No target result may choose a checkpoint, an arm, a seed, an epoch, a crop schedule, a hyperparameter, or a model architecture.

## 5. Pre-training schedule gates for `unet_b1_shuffle`

The shuffle schedule is eligible for pairing interpretation only when it is built and audited before training and meets all conditions for each seed:

1. Every auxiliary image is paired with a mask from a different FLAIR raster (`native_pair_count = 0`).
2. The image and assigned mask remain within the same declared FLAIR region.
3. The raw-ID-10 mask-density multiset is exactly preserved.
4. Mean absolute image-mask density difference is at most `0.005`.
5. The 95th percentile of absolute image-mask density difference is at most `0.05`.

The construction code, schedule JSON, summary statistics, and SHA-256 of each schedule must be archived before training. Every full-budget `unet_b1_shuffle` run must receive its matching precomputed schedule as an input and record its SHA-256. If any gate fails, `unet_b1_shuffle` is recorded as schedule-ineligible and receives no pairing-effect interpretation, regardless of efficacy results. No post-run tolerance is permitted.

## 6. Analytical label-withheld equivalence

No full `B1-nolabel` training arm is planned. In the current experimental family, adding a FLAIR image whose target is all ignore indices is analytically equivalent to `B0-half` for parameter updates in exact arithmetic.

Let `L` be the LoveDA crop selected at update `t` and `U` be any all-ignored auxiliary crop. `loveda_positions_for_update` selects the same LoveDA position `(t,)` for B0-half and B1. The partial-label loss averages only valid pixels. Hence

```
loss(theta; [L, U]) = mean_{p in valid(L)} ell(theta; L, p) = loss(theta; [L]).
```

The frozen encoder stays in evaluation mode, and the calibration-style loss contains no batch-dependent state. Therefore the analytical gradients and optimizer-state transitions are equal. GPU execution may differ by floating-point operation order, so bitwise equality is not claimed.

Before U-Net training, a deterministic gradient unit test must compare a valid crop alone against the same crop batched with an all-ignored crop, using identical initial parameters. It must report maximum absolute loss, gradient, and one-step AdamW parameter differences for every trainable tensor. On CPU, loss and gradients must satisfy `torch.testing.assert_close` with `rtol=1e-6` and `atol=1e-6`; on CUDA, the predeclared tolerance is `rtol=5e-5` and `atol=5e-5` because singleton and batched deterministic convolution kernels can have different reduction order. The observed CUDA loss and gradient differences are retained in the audit JSON. The post-AdamW parameter difference is descriptive: floating-point differences near a zero gradient can be amplified by AdamW's normalization, so it is not required to be bitwise zero. This is an implementation check, not a new efficacy result.

This equivalence rules out an effect of *unlabeled auxiliary-image exposure* on parameter updates in this pure-supervision design. It does not attribute the B1-shuffle residual to label statistics alone: under shuffled supervision, image features, mask geometry, label frequency, and their mismatched interaction remain jointly present.

## 7. Evaluation and estimands

All completed full-budget checkpoints are evaluated on the frozen Paris-excluded 74-city OpenEarthMap population. The primary target metric is rangeland IoU. Secondary metrics are overall mIoU and mIoU over the six leaves with exact LoveDA support, using the original unedited seven-class confusion matrix.

For each arm, the paper will report all three seed-specific metrics, their arithmetic mean, and sample SD. Paired city-bootstrap intervals use the same 2,000 shared city draws and seed `20260803` as the primary study. These intervals must be labelled `checkpoint-conditional`; they are not training-run confidence intervals.

## 8. Predeclared outcomes and interpretation

No effect-size threshold relative to the CLIP probe is used. The following directional classifications are frozen before opening U-Net target metrics:

| Classification | Criterion | Permitted interpretation |
|---|---|---|
| Cross-family recovery direction | Every matched-seed `unet_b1 - unet_b0` rangeland-IoU difference is positive | The declared source mixture recovers the target leaf directionally in both tested model families. |
| Cross-family recovery with mixed seeds | Mean `unet_b1 - unet_b0` rangeland-IoU difference is positive but at least one seed is nonpositive | Report mixed seed behavior; no all-seed directional replication claim. |
| No recovery-direction replication | Mean `unet_b1 - unet_b0` rangeland-IoU difference is nonpositive | Limit the recovery result to the CLIP probe setting. |
| Supported-leaf trade-off direction | Mean matched-seed `unet_b1 - unet_b0` supported-leaf mIoU difference is negative | Report a cross-family trade-off direction, without claiming a common mechanism. |
| Shuffle structure | If schedule gates pass, `R = mean(unet_b1_shuffle - unet_b0) / mean(unet_b1 - unet_b0)` is at least 0.50 | Report that at least half of U-Net recovery remains under the declared mismatched-pair schedule. |

If the denominator of `R` is nonpositive, `R` is not reported. The 0.50 shuffle criterion is a descriptive structural classification, not a statistical significance test. All outcomes, including failed gates, interrupted runs, and non-supportive seeds, remain in the archive and manuscript accounting.

## 9. Execution order and stop rules

1. Create U-Net implementation and deterministic gradient-equivalence test. Run unit tests and a non-evidential smoke run with no more than 20 updates.
2. Freeze implementation manifest, dependency lockfile, pretrained-weight digest, hardware record, and all shuffle schedules.
3. Launch all twelve full-budget runs. A run may be retried only for a documented infrastructure failure before it produces an eligible final checkpoint; both the failed attempt and retry are retained.
4. Run frozen target evaluation and bootstrap scripts for every completed checkpoint. Do not inspect target metrics until all planned arm-seed jobs have terminal statuses.
5. Reconcile metrics with run configurations, generate tables, and run a claim-evidence audit before manuscript revisions.

The protocol is amended only through a new dated file that identifies the changed rule and marks the original run family as not eligible for the amended primary claim.

## 10. Intended paper artifacts

- Main text: one cross-family table reporting seed metrics, mean, SD, and checkpoint-conditional city-bootstrap contrasts.
- Main text or supplement: analytical label-withheld equivalence proposition and its gradient-test result.
- Supplement: U-Net implementation manifest, weight digest, per-run configurations, shuffle schedule audits, attempted/completed run table, and all seed-level metrics.
