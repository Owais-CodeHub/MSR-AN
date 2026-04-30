2D / 3D Radiographic Content-Based Retrieval and Analysis (MSR-AN)

A) Overview

This project implements a medical image retrieval system for 2D scans and pseudo-3D slice ranges using deep feature extraction and similarity matching.

The system:

extracts deep CNN features
aggregates slice features using LSTM-compatible formatting
performs cosine similarity matching
retrieves Top-K similar scans
displays results in an interactive MATLAB GUI

Supported retrieval modes:

2D retrieval → single slice
3D retrieval → multiple consecutive slices
Software Requirements

Tested with:

MATLAB R2024a

Required toolboxes:

Deep Learning Toolbox
Image Processing Toolbox
Computer Vision Toolbox
Statistics and Machine Learning Toolbox
Parallel Computing Toolbox (optional GPU acceleration)
Required Model Files

Place these files in the same folder as:

mainCode.m

Required:

model_1_MSA_SN.mat
model_2_RFA_SN.mat
Folder Structure

Expected structure:

Project Folder

mainCode.m
model_1_MSA_SN.mat
model_2_RFA_SN.mat

Query DB/
    Infected/
    Normal/

Retrieval DB/
    Infected/
    Normal/

Each folder must contain 2D scan images.

How to Run

Step 1

Open MATLAB R2024a

Step 2

Open project folder

Step 3

Open:

mainCode.m

Step 4

Run:

Press F5

or click:

Run

MATLAB will execute retrieval and automatically display GUI results.

Selecting Query Scans

Inside:

mainCode.m

Locate:

queryImg = 14:15;

Examples:

Single slice (2D retrieval):

queryImg = 14;

Multiple slices (pseudo-3D retrieval):

queryImg = 14:15;

System automatically averages slice features.

Retrieval Method

Processing pipeline:

Input Query Scan
→ Deep Feature Extraction
→ Feature Alignment for LSTM
→ Sequence Feature Aggregation
→ Cosine Similarity Matching
→ Top-K Retrieval
→ GUI Visualization
Default Hyperparameters

These are MATLAB default settings used in training models.

Dense Feature Extraction Block
Input size: 224 × 224 × 3
Feature layer: dropout
Activation layer: dropout7
Normalization: batch normalization
Pooling: global average pooling
LSTM Feature Aggregation
Sequence length: 16
Feature alignment: equal-sample sequence formatting
Temporal aggregation: sliding window sequence embedding
Similarity Retrieval
Similarity metric: cosine similarity
Top-K matches: 5
Feature fusion: mean aggregation across slices

Modify:

topK = 5;

to change retrieval count.

GPU Support (Optional)

If GPU available:

gpuDevice(1)

Otherwise remove:

gpuDevice(1);
reset(gpuDevice());

Code will run on CPU normally.

Dataset Note

This project uses a public medical dataset referenced in the paper.

Due to dataset sharing restrictions:

Full dataset cannot be redistributed

Therefore:

Sample query and retrieval images are included only for demonstration.

To reproduce full results:

Download dataset from link provided in the paper.

Output GUI Displays

The MATLAB GUI shows:

Query scan (2D or 3D slice range)
Predicted class label
Actual class label
Top-K retrieved scans
Similarity score per retrieval
Similarity score bar chart
Summary panel
Citation

If you use this code in research, please cite:

MSR-AN by Owais et al.
2D / 3D Radiographic Content-based Retrieval and Analysis

If you'd like, I can also generate a one-page version suitable for supplementary material submission with your paper/code release (many journals require that format).