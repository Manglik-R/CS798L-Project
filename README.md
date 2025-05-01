# CS798L Project: Differentially Private N-gram Extraction

This repository contains a clean implementation of the Differentially Private N-gram Extraction (DPNE) algorithm, as proposed in [Kim et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/45feb0d2b1a80f14239a2a8055f476af-Paper.pdf). The DPNE algorithm allows for the extraction of frequent n-grams from user text data while satisfying rigorous differential privacy (DP) guarantees.

## Repository Structure

```
.
├── DPNE.py                 # Main DPNE algorithm
├── DPSU.py                 # Differentially Private Set Union (DPSU) module
├── util.py                 # Helper utilities (text cleaning, validation, etc.)
├── experiments.ipynb       # Analysis and experiments with different datasets
├── requirements.txt        # Python dependencies
├── experiment_scripts/
│   ├── msnbc.py            # MSNBC dataset pipeline
│   └── topical_chat.py     # Topical Chat dataset pipeline
├── Result/
│   ├── n_grams/            # Pickled extracted n-gram sets
│   └── plots/              # Visualizations of n-gram statistics
├── CS798L_Project_Report   # Report
└── README.md               # This file
```

## Overview
- Implements the complete DPNE pipeline.
- Supports any dataset
- Saves extracted n-grams (`.pkl`) and corresponding plot visualizations (`.png`)

## Usage

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Running the DPNE Algorithm

You can run the DPNE module directly or through provided scripts for specific datasets:

```python
from DPNE import DPNE

# Example usage
user_texts = ["this is an example", "differential privacy is cool"]
dpne = DPNE(user_texts, epsilon=1.0, delta=1e-5, p=0.01, T=3)
dpne.run("example_run")
```

For real datasets, use the provided scripts by tuning hyperparameters:

```bash
python experiment_scripts/msnbc.py
python experiment_scripts/topical_chat.py
```

### 3. Analyze Results

Open the `experiments.ipynb` to view results, visualize n-gram frequencies, and compare across datasets.

## Output

- `Result/n_grams/` contains pickled files with the extracted n-gram sets for each value of `k`.
- `Result/plots/` contains the line plots showing the number of k-grams discovered at each iteration.

## Datasets
Datasets can be downloaded from:
- **MSNBC Dataset** — Short browser histories. [Link](https://archive.ics.uci.edu/dataset/133/msnbc+com+anonymous+web+data).
- **Topical Chat Dataset (Amazon)** — Long-form user dialogues. [Link](https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat).
- **LMSYS Dataset** — Hugging Face LLM chat data. [Link](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).


Refer to the original paper for theoretical guarantees.

For more information regarding implementation, please refer to the Report.

## References

- Kim et al., "[Differentially Private N-gram Extraction](https://proceedings.neurips.cc/paper_files/paper/2021/file/45feb0d2b1a80f14239a2a8055f476af-Paper.pdf)", NeurIPS 2021.