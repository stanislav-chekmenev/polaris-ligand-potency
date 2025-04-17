# Polaris Potency Prediction Challenge

## Overview

This repository contains the solution to the Polaris Potency Prediction Challenge. The goal is to predict antiviral potencies for two specific virus targets: **SARS-CoV-2 Mpro** and **MERS-CoV Mpro**.

Our solution integrates several advanced machine learning techniques and is primarily inspired by the [ConAN](https://github.com/duyhominhnguyen/conan-fgw/tree/main) approach, which introduces a novel conformer aggregation method using **Fused Gromov-Wasserstein barycenters**.

### Key Modifications to the Original ConAN Method:

- **SO(3)-Equivariant GNN**: We utilize [MACE](https://github.com/ACEsuit/mace/tree/main), an advanced equivariant graph neural network, for robust 3D molecular feature extraction.
- **Molecular Embeddings**: Strong, chemically-informed embeddings are extracted using the [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry) model.
- **Transformer Integration**: An additional transformer block has been introduced to effectively combine and weight information from diverse molecular embedding sources, enhancing predictive performance.

## Installation

We recommend using either **Conda** or **Micromamba** to manage dependencies. Install all necessary packages via the provided environment file:

```bash
conda env create -f env.yml
# or
micromamba create -f env.yml
```

Activate the environment after installation:

```bash
conda activate polaris-env
# or
micromamba activate polaris-env
```

## Notebooks

The `notebooks` directory contains resources including:

- A **baseline notebook** demonstrating a simple, non-deep learning method for potency prediction.
- A dedicated notebook illustrating the core concepts and practical use of the **MACE architecture**.

## Results and Ongoing Development

Final results and essential model refinements are currently under development. Initial experiments indicate rapid overfitting to the training set, prompting ongoing optimization efforts. Additionally, we have established a strong baseline using an MLP model with ChemBERTa-derived features, which has proven challenging to surpass thus far.

