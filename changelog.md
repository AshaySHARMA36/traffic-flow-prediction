# Changelog

All notable changes to the "Vision-Based Traffic Flow Prediction" project will be documented in this file.

## [Unreleased]

## [Day 3] - 2025-12-02
### Added
- **Computer Vision Pipeline:** Implemented `VideoProcessor` (resizing/normalization) and `VideoAugmentor` (Albumentations).
- **Model Architecture:** Built `SpatialEncoder` with Dual-Stream support (ResNet18 + Custom 8-channel CNN).
- **Data Engineering:** Created `TrafficVideoDataset` and `DataLoader` with lazy loading.
- **Optimization:** Added `preprocess_videos.py` for offline resizing (achieved 15x speedup).
- **Sampling:** Implemented Sliding Window and Adaptive sampling strategies.
- **Configuration:** Added `preprocessing_config.yaml` and integration tests.

## [Day 2] - 2025-12-01
### Added
- **Data Engineering:** Created `src/utils/organize_datasets.py` to standardize CityFlow (video) and Traffic4Cast (heatmap) files.
- **Exploratory Data Analysis (EDA):** Implemented `src/utils/eda_videos.py` to analyze resolution, lighting, and traffic density distributions.
- **Data Splitting:** Created `src/data/make_splits.py` to generate strict temporal train/val/test splits (`configs/data_splits.json`).
- **Documentation:** Added `dataset_stats.md` detailing dataset characteristics and preprocessing strategy.
- **Verification:** Added `src/utils/test_datasets.py` to validate file integrity and metadata.

## [Day 1] - 2025-11-30
### Added
- **Project Structure:** Established standard ML directory layout (`src`, `configs`, `data`, `experiments`).
- **Environment:** Created `requirements.txt` with PyTorch 2.1 (CUDA 11.8) and OpenCV 4.8.
- **Data Pipeline:** Implemented `VideoProcessor` class in `src/utils/video_processor.py`.
- **Documentation:** Initialized README and git configuration.