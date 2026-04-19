# ParkSense Replit Setup

This project can be moved to Replit with ZIP import.

## What to include in your ZIP

Keep these folders and files:

- `parksense/`
- `artifacts/`
- `data/`
- `pipeline.py`
- `finalize_spiral_models.py`
- `screen_user.py`
- `prepare_handwriting_dataset.py`
- `requirements.txt`

Do not include:

- `venv/`
- `__pycache__/`

## Replit import steps

1. Open `https://replit.com/import`
2. Choose `ZIP`
3. Upload your project ZIP
4. Wait for Replit to finish importing
5. Open the Shell tab
6. Run `pip install -r requirements.txt`

## First commands to run on Replit

Train or refresh spiral artifacts:

```bash
python finalize_spiral_models.py
```

Run a spiral-only test:

```bash
python screen_user.py --spiral-image-path /absolute/or/replit/path/to/image.png
```

Run a combined test with placeholder voice and tapping scores:

```bash
python screen_user.py --spiral-image-path /absolute/or/replit/path/to/image.png --voice-score 0.71 --tapping-score 0.58
```

## What score to use later

When you combine models, use:

- `spiral_prediction.recommended_multimodal_score`

Only use it if:

- `spiral_prediction.multimodal_ready` is `true`

If it is `false`, ask the user to redraw the spiral or rely on the other modalities.
