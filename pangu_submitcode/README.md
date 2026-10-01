# Reviewer Hard-Mode Bundle (Paper-Aligned)

This folder is intentionally organized to map to the paper pipeline while remaining manual-heavy.

## Paper-to-Code Mapping

- **Section 4.2 Global Weather Model Prediction**
  - `data_prepare.py`
  - `inference.py`
  - `forecast_decode.py`
- **Section 4.3 Physics Distillation Module (PDM)**
  - `pdm_manual_pack.py`
- **Section 4.4 GRCMI-WPF Training**
  - `train_grcmi_wpf.py`

## Hard-Mode Execution

Run from repository root:

```bash
bash pangu_submitcode/run_reviewer_pipeline.sh stage1
```

```bash
export DATASET_TAG=lhb_2017_2020
export UPPER_RAW=/absolute/path/to/upper_raw.npy
export SURFACE_RAW=/absolute/path/to/surface_raw.npy
export TIME_LABELS=/absolute/path/to/time_labels.npy
export PACK_OUTPUT_DIR=/home/ike/WY/pangu/data/processed
bash pangu_submitcode/run_reviewer_pipeline.sh stage2
```

```bash
export TRAIN_CONFIG=pangu_submitcode/config/configGRCMI_WPF_submit_WF3.json
export TRAIN_OUTPUT_ROOT=pangu_submitcode/experiments/reviewer_hardmode
bash pangu_submitcode/run_reviewer_pipeline.sh stage3
```

## Output Files (stage3)

- `all_epochs_metrics.csv`
- `best_epoch_test_results.csv`
- `best_model.pth`
- `config_snapshot.json`
- `train.log`

## Pangu Model Downloads

Source repo: https://github.com/198808xc/Pangu-Weather

- `pangu_weather_1.onnx`
  - Google Drive: https://drive.google.com/drive/folders/1fYRE3L4NvuFU6fP41fJ6xF0zaMI8HzIa?usp=drive_link
  - Baidu Netdisk: https://pan.baidu.com/s/1oCCJ8bZsY3B4u7-4Yk8PPA?pwd=iw9u
- `pangu_weather_3.onnx`
  - Google Drive: https://drive.google.com/drive/folders/1la2pV5w286Jm9k8j6fVwVYLrVvLwOq8v?usp=drive_link
  - Baidu Netdisk: https://pan.baidu.com/s/10bHV7Myt2Nz40n4W8YO_cA?pwd=5e7h
- `pangu_weather_6.onnx`
  - Google Drive: https://drive.google.com/drive/folders/1FblM6Qxq7ukEjf0h7EGF6Nlfv6wVYhE8?usp=drive_link
  - Baidu Netdisk: https://pan.baidu.com/s/1QfR8f_yE4f33R5Qe2YA6wA?pwd=4bcw
- `pangu_weather_24.onnx`
  - Google Drive: https://drive.google.com/drive/folders/1z8_mJwpXXwksVjByWwTRf6t1f7h0M9rM?usp=drive_link
  - Baidu Netdisk: https://pan.baidu.com/s/1I8Vj6TqSBX3GJwAW6rGY5w?pwd=2vhn
