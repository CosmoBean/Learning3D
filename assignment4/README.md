# 16-825 Assignment 4: Multi-View Transformers

Goals: In this assignment, you will prepare a multi-view dataset, synthesize novel views with a mini-LVSM model, and reconstruct 3D geometry from images with a mini-VGGT model.

## Setup

Run all assignment commands from the repository root. Install PyTorch for your machine, then install the assignment dependencies:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```



### Dataset

We use prepared iPhone RGB/depth frames from [ScanNet++](https://github.com/scannetpp/scannetpp). The supplied subset contains 280 indoor scenes, split by scene into 224 training, 28 validation and 28 test scenes. Each scene contains paired RGB/depth frames and camera annotations. Training RGB images and aligned depth maps are 400×300. Depth maps are 16-bit PNGs in millimeters, with zero values indicating missing depth.

Download and extract [ScanNetpp_A4.zip](https://huggingface.co/datasets/learning3dvision/ScanNetpp_iphone/blob/main/ScanNetpp_A4.zip) once for training, validation and test (31.9 GB compressed; 40.7 GB extracted):

```bash
pip install huggingface_hub
hf download learning3dvision/ScanNetpp_iphone ScanNetpp_A4.zip --repo-type dataset --local-dir data/scannetpp
unzip data/scannetpp/ScanNetpp_A4.zip -d data/scannetpp
```

The training loader applies the preprocessing functions you will implement to data in `processed/`. Validation and test data are already prepared in `prepared_eval/` (128 frames per sequence) and use the provided evaluation loader.

## Assignment parts

1. **[Multi-View Data Preprocessing](docs/part1_data.md) (20 points)** — Prepare canonicalized cameras, depth, points and ray embeddings. Complete Q1 and check your geometry.
2. **[Novel-View Synthesis with Mini-LVSM](docs/part2_lvsm.md) (40 points)** — Predict target RGB from source images and camera rays. Complete Q2 and evaluate the ablations.
3. **[Multi-View Geometry with Mini-VGGT](docs/part3_vggt.md) (40 points)** — Predict cameras and depth from RGB views. Complete Q3, overfit fixed clips, then train and evaluate.

The [data contract](docs/DATA_CONTRACT.md) defines shared tensor shapes and coordinate conventions. Each part includes its TODOs, commands, metrics and submission requirements.

## Submission

Present your results on your assignment webpage, organized into the three parts. Include the requested visualizations, training curves, evaluation metrics and explanations listed in each part's submission block.
