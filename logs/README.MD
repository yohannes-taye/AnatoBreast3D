# Logs

Generated log files are local runtime artifacts and are ignored by git.

- `training/`: logs from `tools/train_breastseg.py`, for example `train_breastseg_b2_*.log`.
- `orbbec_sdk/`: Orbbec SDK hardware logs, for example `OrbbecSDK.log*.txt`.

The Orbbec SDK may still create a capitalized `Log/` folder when hardware code
runs from the project root. Move those files here when you want the project tree
to stay tidy.
