# Multimodal Graph-Based Mistake Detection in Procedural Activities

A research project investigating procedural error detection in egocentric cooking videos through supervised baselines and graph-based multimodal learning. Developed as part of the Advanced Machine Learning course at Politecnico di Torino.

## Overview

Detecting procedural errors in egocentric videos requires understanding both visual observations and their semantic relationship to procedural knowledge encoded in textual instructions. This work addresses this challenge through two complementary tasks:

**Task 1: Supervised Error Recognition** - Reproduction of Captain Cook 4D benchmark baselines (MLP, Transformer, LSTM) for binary video classification detecting visually obvious errors.

**Task 2: Multimodal Graph-Based Extension** - A novel framework using Directed Acyclic Graph Neural Networks (DAGNNs) that models recipe structure while integrating visual and textual information through Hungarian matching.

We evaluate on the **Captain Cook 4D** dataset containing 384 egocentric cooking videos across 24 recipes with procedural error annotations.

## Motivation

Procedural error detection faces three fundamental challenges:

1. **Semantic Gap**: Low-level visual features must be mapped to high-level procedural concepts in recipe instructions
2. **Temporal Alignment**: Variable-length observed actions must be aligned with prescribed recipe steps
3. **Causal Structure**: Actions can only be identified as erroneous within their broader procedural context

Traditional action recognition classifies isolated actions without considering procedural context or textual instructions. Our graph-based approach explicitly models recipe structure as directed acyclic graphs (DAGs) where steps must be executed in specific causal order, while integrating multimodal visual and textual embeddings.

## Approach

### Task 1: Supervised Baselines

We reproduce and extend the Captain Cook 4D benchmark with three temporal aggregation architectures operating on pre-extracted visual features:

- **MLP**: Multi-Layer Perceptron with LayerNorm, 512 hidden units, dropout 0.5
- **Transformer**: 2 encoder layers, 8 attention heads, sinusoidal positional encoding  
- **LSTM**: 2 stacked layers, 1024 hidden size, using final hidden state for classification

### Task 2: Graph-Based Multimodal Pipeline

Our multimodal framework consists of four stages:

1. **Feature Extraction**: EgoVLP encoders generate 256-dim visual embeddings from 1-second video clips and textual embeddings from recipe step descriptions

2. **Visual Step Segmentation**: HiERO performs zero-shot hierarchical localization via agglomerative clustering, producing visual step proposals (768-dim after HiERO projection)

3. **Multimodal Alignment**: Hungarian algorithm finds optimal bipartite matching between visual and textual step embeddings, minimizing cosine distance

4. **Graph-Based Classification**: DAGNN processes directed graphs where nodes represent recipe steps with concatenated visual+text features (1536-dim), edges encode sequential dependencies, and GCN layers propagate procedural context for video-level error detection


## Installation

### Prerequisites
- Python 3.12
- CUDA 12.6 (for GPU acceleration)
- ~10GB disk space for dependencies and dataset

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/MarcoPernoVDP/MistakeDetection.git
cd MistakeDetection
```

2. Create and activate virtual environment:
```bash
# Install virtualenv
pip install virtualenv

# Create environment
virtualenv venv --python=3.12

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

3. Install PyTorch with CUDA support:
```bash
pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
```

4. Install remaining dependencies:
```bash
pip install -r requirements.txt
```