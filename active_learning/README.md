# Active Learning Annotation Studio

This folder stores the live annotation loop used by `viewer/live.html`.

Generated data is written to `active_learning/dataset` using the existing
BreastSeg training layout:

- `images/*.png`
- `sem_masks/*.png`
- `annotations/coco_det.json`
- `splits/train.txt`
- `splits/val.txt`

`active_learning/configs/breastseg_active.py` is a short fine-tuning config that
reuses the current MammaSegNet training code.
