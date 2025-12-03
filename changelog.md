# Changelog

All notable changes to the "Vision-Based Traffic Flow Prediction" project will be documented in this file.

## [Unreleased]
## [Day 7] - 2025-12-06
### Added
- **Global Temporal Modeling:** Implemented `TemporalTransformer` to capture long-range dependencies using Multi-Head Self-Attention.
- **Attention Mechanism:** Built `ScaledDotProductAttention` and `MultiHeadAttention` modules from scratch.
- **Positional Encoding:** Implemented Sinusoidal Positional Embeddings to preserve sequence order in parallel processing.
- **Visualization:** Added tools to visualize attention maps (`visualize_multi_head_attention`) to debug model focus.
- **Architecture:** Stacked Transformer Encoder blocks with Pre-Norm architecture for stable training.

## [Day 6] - 2025-12-05
### Added
- **Temporal Encoder:** Implemented `TemporalConvNet` (TCN) with dilated residual blocks.
- **Benchmarking:** Analyzed TCN performance vs LSTM. TCN is **6x smaller** (0.3M vs 1.8M params) and faster.
- **Integration:** Created `SpatioTemporalEncoder` to fuse ResNet50 spatial features with TCN temporal features.
- **Architecture:** Validated exponential dilation (1, 2, 4) achieving 15-frame receptive field.

## [Day 5] - 2025-12-04
### Added
- **Transformer Model:** Implemented `TrafficTransformer` in `src/models/transformer.py` for global temporal context.
- **Attention Mechanism:** Integrated PyTorch's `TransformerEncoderLayer` for Multi-Head Self-Attention.
- **Positional Encoding:** Added sinusoidal embeddings to preserve temporal order in the video sequence.
- **Testing:** Verified Transformer forward pass and gradient stability with ResNet backbone.

## [Day 4] - 2025-12-03
### Added
- **Baseline Model:** Implemented `ConvLSTMCell` and `TrafficFlowPredictor` for spatiotemporal forecasting.
- **Training Pipeline:** Created `src/train_baseline.py` with training, validation, and early stopping loops.
- **Metrics:** Implemented MAE, RMSE, MAPE, and R2 score tracking.
- **Architecture:** Designed a modular ConvLSTM system capable of stacking multiple layers.

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