# Location Generalization — Core Implementation (Anonymized)

Core implementation accompanying ICLR 2027 submission 22439, "Taming Continuous
Location Tokens for Generalized Next Point-of-Interest Recommendation."

This release showcases the implementation; it contains the model, training,
evaluation and protocol code paths referenced in the paper and rebuttal. It is
a code reference, not a packaged runnable distribution.

| Path | Contents |
|---|---|
| `models/location_encoder.py` | All location encoders: fixed/trainable RFF, unit-sphere RFF, Space2Vec grid/hex, Sphere2Vec-sphereC, per-POI discrete, H3, OSM-region. |
| `models/latlon_transformer.py` | Causal transformer over location embeddings. |
| `utils/strict_unseen.py` | Strict protocol: coordinate holdouts per H3 cell, user splits, tie-aware evaluation (expected-random tie metrics), candidate vocabularies. |
| `utils/evaluation.py` | Ranking metrics (Recall@k, MRR) with seen/unseen and first-visit groups. |
| `utils/latlon_dataset.py` | Check-in sequence dataset and collation. |
| `scripts/strict_unseen_train.py` | Training and frozen-checkpoint evaluation entry point with artifact/hash verification. |
| `scripts/rebuttal_extended_eval.py` | Extended metrics evaluation (R@5/10/20/100, MRR, expected NDCG@10). |
| `training/` | Encoder construction and the full argument surface. |
| `location_generalization/` | Experiment CLI: declarative sweep configs, validation-based range selection, final evaluation. |
| `configs/unseen/` | The exact sweep configurations used for the paper and rebuttal experiments (per-encoder wavelength ranges, seeds, splits). |

Data: MASSIVE-STEPS check-ins (public; see the dataset's HuggingFace page).
