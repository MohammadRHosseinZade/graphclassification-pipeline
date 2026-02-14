# Graph Classification Pipeline

A complete machine learning pipeline for **graph model classification** using structural network features. This project generates synthetic graphs from three random graph models, extracts topological features, and trains multiple ML classifiers to distinguish between them.

The pipeline covers: graph generation, feature engineering, preprocessing, model training, validation, and visualization.

---

# Overview

This repository builds and classifies graphs from three models:

* Erdős–Rényi (ER)
* Watts–Strogatz (WS)
* Barabási–Albert (BA)

From each graph, 21 structural features are extracted and used to train:

* k-Nearest Neighbors
* Random Forest
* Support Vector Machine

All models achieve perfect classification on the generated dataset.

---

# Installation

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

Install required packages:

```bash
pip install networkx numpy pandas scipy scikit-learn matplotlib seaborn jupyter tqdm
```

---

# Jupyter Notebook Version

An interactive notebook version of the full pipeline is included:

```
notebook.ipynb
```

Run with:

```bash
jupyter notebook
```

Open the notebook and run all cells to reproduce:

* Graph generation
* Feature extraction
* Scaling
* Model training
* PCA and t-SNE visualization
* Evaluation and plots

---

# Dataset Generation

## Graph Counts

* Total graphs: **150**
* Per model: **50**
* Nodes per graph: **150**

## Model Parameters

### ER Graphs

* Parameter: edge probability `p ~ Uniform(0.02, 0.15)`

### WS Graphs

* k ∈ {4, 6, 8, 10}
* beta ~ Uniform(0.05, 0.4)

### BA Graphs

* m ∈ {2, 3, 4, 5}

## Storage Format

Graphs are saved as NetworkX pickles:

```
graphs/
├── ER/
├── WS/
└── BA/
```

Each folder contains 50 `.gpickle` files.

---

# Feature Extraction

## Structural Features (21 Total)

Includes:

* Nodes, edges, density
* Radius, diameter
* Average degree, degree variance, max degree
* Global & average clustering
* Freeman centralization
* Avg betweenness, closeness, pagerank
* Degree entropy
* Assortativity
* Avg shortest path
* Connectivity measures
* Spectral radius

## Notes

* Distance metrics computed on largest connected component
* Exact algorithms used (no approximation)
* Parallel feature extraction supported
* Robust handling of disconnected graphs

Output:

```
graph_features.csv
graph_features_scaled.csv
```

---

# Preprocessing

* Z-score normalization (StandardScaler)
* Scaler fit only on training data
* Stratified train/test split

## Split

* Train: 140 graphs
* Test: 10 graphs
* Random seed: 42

Files:

```
train.csv
test.csv
```

---

# Machine Learning Models

## kNN

* k = 3
* Distance metrics:

  * Euclidean
  * Manhattan
  * Cosine

## Random Forest

* 100 trees
* random_state = 42

## SVM

* RBF kernel
* random_state = 42

## Metrics Reported

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-score (weighted)
* Confusion matrices

Results saved:

```
ml_results.csv
```

---

# Validation

A validation script verifies correctness:

```
validate_split.py
```

Checks include:

* No train/test leakage
* Disjoint splits
* Proper scaler fitting
* Stratified class balance
* 5-fold cross-validation

Cross-validation confirms consistent perfect accuracy across folds.

---

# Dimensionality Reduction

## PCA

* Components: 3
* ~93% variance explained
* Feature loading analysis included

Outputs:

```
pca_results.csv
plots/pca_2d.png
plots/pca_variance.png
```

## t-SNE

* 2D embedding
* Perplexity = 30
* max_iter = 1000

Output:

```
plots/tsne_visualization.png
```

---

# Visualizations

Saved under `plots/`:

* Feature distributions
* Class-wise boxplots
* Correlation heatmap
* PCA variance chart
* PCA 2D projection
* t-SNE embedding
* kNN confusion matrices
* Model comparison charts

---

# Main Scripts

```
pipeline_script.py     # Full pipeline script
notebook.ipynb         # Notebook version
validate_split.py      # Validation checks
```

Run the full pipeline:

```bash
python pipeline_script.py
```

---

# Outputs

## Graph Files

```
graphs/ER/*.gpickle
graphs/WS/*.gpickle
graphs/BA/*.gpickle
```

## Data Files

```
graph_features.csv
graph_features_scaled.csv
train.csv
test.csv
```

## Results

```
ml_results.csv
pca_results.csv
```

## Plots

```
plots/*.png
```

---

# Key Findings

* Structural features clearly separate ER, WS, and BA graphs
* Multiple ML models reach perfect performance
* Feature space is highly discriminative
* PCA shows strong low-dimensional structure
* t-SNE shows clean class clusters

---

# Reproducibility

All random processes use:

```
random_state = 42
```

to ensure reproducible results across runs.

---

# License / Usage

Free to use for coursework, experiments, and research. Cite your work appropriately if used in academic submissions.
