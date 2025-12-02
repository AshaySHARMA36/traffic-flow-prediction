Dataset Characteristics & Statistics

Project: Vision-Based Traffic Flow Prediction

Date: Day 2 Analysis

1. Traffic4Cast (London Subset)

Data Type: Spatio-Temporal Traffic Heatmaps (HDF5 format)

Total Files: 109 (Daily Movie Files)

Temporal Resolution: 5-minute bins (288 time steps per file/day)

Spatial Resolution: 495 $\times$ 436 grid cells

Channels: 8 (Volume/Speed for N, NE, E, SE directions)

Total Duration Covered: 109 Days

Challenges: - Sparsity: Many grid cells are zero (no roads/traffic).

Dimensionality: 8 channels is non-standard for pre-trained vision models (requires custom input layer).

Non-Visual: Lacks texture/object features; relies purely on flow dynamics.

2. CityFlow (AI City Challenge 2022)

Data Type: Raw RGB Video Footage

Total Videos: 36 (Renamed cityflow_001.avi to cityflow_036.avi)

Resolution: $1920 \times 1080$ pixels (Full HD)

Frame Rate (FPS): 10.0 FPS

Average Duration: ~97.4 seconds per clip

Dominant Condition: Daytime, High Traffic Density (25/36 videos).

Challenges:

High Resolution: 1080p is too large for Transformers; requires downsampling.

Lighting Bias: Dataset is 100% daytime; requires augmentation to generalize to night.

Perspective: Camera angles vary significantly between Scenarios (S01, S03, S04).