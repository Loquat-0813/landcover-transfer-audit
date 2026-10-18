# Land-cover transfer audit

Code and retained analysis records for **Auditing Source-Label Coverage and Recovery–Retention Trade-offs in Cross-Dataset Land-Cover Segmentation**.

This study audits source-mask label support and describes recovery–retention trade-offs under specific source-mixture comparisons. It does not propose a successful mitigation algorithm or claim that label coverage alone causes the observed differences. OpenEarthMap rangeland includes shrubs and is an operational vegetation proxy, not a pure-herbaceous reference.

## Contents

- `p1data/`, `p1backbone/`, `p1model/`, `p1train/`, `p1eval/`, `scripts/`: retained dataset, training, evaluation and numerical-analysis tools. Some shared modules implement historical exploratory variants; their presence is not evidence of a successful paper contribution.
- `ontology/`, `configs/`, `protocols/`: mappings, settings and retained protocol records.
- `archive/`, `artifacts/audit_submission_20261004/`, `outputs/`, `audits/`, `evidence/`: retained integer confusions, identifiers, metrics, configurations and audit summaries. Original relative provenance paths are preserved. Paths inside historical JSON may refer to the original workstation or GPU host; supply your own dataset/model paths when running training or evaluation.
- `artifacts/paper_positioning_20261015/recompute_native_unet.py`: recomputes the native U-Net summary and city-bootstrap contrasts from retained numerical records, without model inference.
- `paper/english/`, `paper/chinese/`: 40-reference manuscript sources and PDFs; Chinese editable Markdown is also included. Original figure interiors and formal reference titles remain in English.
- `docs/`: evidence boundaries, citation checks and publicly attributed journal-ranking sources.
- `SHA256SUMS.json` and `verify_files.py`: published-file byte checks.

## Numerical checks

Use Python 3.12 with dependencies from `requirements-analysis.txt`. From the repository root:

```bash
python verify_files.py
python -m unittest -v tests.test_fixed_budget tests.test_semantic_drone_bootstrap tests.test_supported_leaf_bootstrap tests.test_qualitative_selection
python artifacts/paper_positioning_20261015/recompute_native_unet.py
```

The recomputation command writes `NATIVE_UNET_RECOMPUTED.json`; it does not train or infer. The manifest checks exact published bytes, so verify before regenerating derived outputs. Successful numerical checks verify only these bounded procedures.

## Training and data

Training/evaluation requires the original **LoveDA, FLAIR, OpenEarthMap and Semantic Drone** releases, the specified CLIP backbone, a compatible CUDA/PyTorch environment and experiment-specific path arguments. Historical entry points include `scripts/train_loveda_e0_e1.py`, `scripts/train_loveda_flair_b1.py`, `scripts/train_loveda_flair_unet.py` and the corresponding evaluation tools. `requirements-e1.txt` records part of the training dependencies; it is not a complete historical environment lock. Consult each tool's arguments and retained run configurations. This publication has not rerun model training or pixel-level evaluation.

Provider RGB imagery and spatial masks, foundation-model weights, recovered model checkpoints, credentials, SSH control files and third-party source trees are not distributed here. Obtain provider data and pretrained models under their original terms. The nine recovered native U-Net checkpoints remain in the authors' separate local archive; no public checkpoint availability is claimed.

## Evidence limits

The original main pairing schedules and two Semantic Drone per-checkpoint raster records were not recovered. Image and label sources change together. Model families share evaluation targets. Historical file-hash matching does not prove initialization identity, historical execution provenance, pixel-mask identity or a fresh inference replay. Bootstrap intervals are conditional on the retained checkpoints and sampled evaluation units, not complete uncertainty across training runs. See `docs/EVIDENCE_LIMITS.md` and the manuscript.

The 2026-10-16 citation check is a provisional assistant review: 12 new bilingual entries were checked, with exact-context bridges for the earlier English references. The full Chinese text was translated and mechanically checked by the preparing assistant; the remaining Chinese citation contexts were not all freshly reviewed independently. Public 2025 CAS large-category ranking transcriptions are attributed in `docs/`; authenticated official journal-by-journal ranking records were unavailable.

## Manuscript compilation

Run `pdflatex`, BibTeX and two further `pdflatex` passes on `paper/english/main.tex` (and `supplement.tex`). Use XeLaTeX and BibTeX for `paper/chinese/main.tex`; its CTeX Windows font setting may require an appropriate locally available font setting on other platforms. Original template notices are preserved.

No manuscript has been submitted or accepted through this repository, and no archival DOI has been assigned. Do not interpret a public repository as a claim that all historical evidence or the complete experiment environment has been recovered. No project-wide reuse license has yet been selected by the authors; upstream template notices remain applicable.
