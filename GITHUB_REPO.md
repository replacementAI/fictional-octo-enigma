# GitHub Repo Guide

## What to put in the repo

### Python scripts

- `parksense/`
- `pipeline.py`
- `finalize_spiral_models.py`
- `screen_user.py`
- `prepare_handwriting_dataset.py`
- `train_handwriting_image_model.py`
- `compare_handwriting_models.py`
- `analyze_spiral.py`
- `requirements.txt`
- `REPLIT_SETUP.md`

### Small training data you may include privately

- `data/Spiral_HandPD.csv`
- `data/Meander_HandPD.csv`

### Generated artifacts you can include if you want the repo to run immediately

- `artifacts/spiral_pipeline.joblib`
- `artifacts/spiral_image_pipeline_final.joblib`

## What not to upload publicly by default

- `venv/`
- `__pycache__/`
- `archive.zip`
- `data/archive.zip`

## Important licensing note

The UCI voice dataset is listed under CC BY 4.0.

The augmented handwriting Kaggle dataset `sowmyabarla/parkinsons-augmented-handwriting-dataset` is listed as CC0 on Kaggle.

The `claytonteybauru/spiral-handpd` Kaggle mirror shows `License: Unknown` on Kaggle. Because of that, the safest public GitHub setup is:

1. Upload the code
2. Upload only the data you are comfortable sharing privately
3. For a public repo, prefer adding download instructions instead of redistributing unknown-license raw files

## Safest public repo pattern

Public repo:

- code
- `requirements.txt`
- model scripts
- README/setup docs
- optional trained artifacts

Private repo:

- everything above
- `data/Spiral_HandPD.csv`
- `data/Meander_HandPD.csv`
- optional `augmented_hw_dataset/`

## Commands after pushing

```bash
python finalize_spiral_models.py
python screen_user.py --spiral-image-path path/to/spiral.png --voice-score 0.71 --tapping-score 0.58
```
