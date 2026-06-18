# Data Layout

The project separates data-loading code from local training data:

```text
data_loaders/                  # Python package imported by training code
  breast_coco.py               # BreastCocoDataset and breast_collate_fn

data/breastseg/                # local data, ignored by git
  raw_combined/
  dataset3_converted/
  processed_merged/
    images/
    sem_masks/
    annotations/coco_det.json
    splits/train.txt
    splits/val.txt
```

The default training config points to:

```text
./data/breastseg/processed_merged
```

The loader import path is:

```python
from data_loaders.breast_coco import BreastCocoDataset, breast_collate_fn
```

To rebuild a processed dataset from a zip:

```bash
python tools/prepare_data.py \
  --zip /path/to/BreastSegDataset.zip \
  --out ./data/breastseg/processed_merged
```

To convert LabelMe-style annotations:

```bash
python tools/convert_breast_labelme.py \
  --json-dir ./data/breastseg/raw_combined/labelme \
  --image-dir ./data/breastseg/raw_combined/images \
  --out ./data/breastseg/processed_merged
```

To regenerate train/val splits:

```bash
python tools/regenerate_splits.py \
  --images-dir ./data/breastseg/processed_merged/images \
  --splits-dir ./data/breastseg/processed_merged/splits \
  --dataset3-images-dir ./data/breastseg/dataset3_converted/images \
  --dataset3-list ./data/breastseg/processed_merged/dataset3_files.txt
```
