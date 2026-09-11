# 🛡️ Cyberbullying Tweet Classification

[IT Versione italiana](README.it.md)

**Machine Learning & NLP project developed for the Intelligent Systems course at the University of Bari (2026)**

**Authors:** Daniele Spinelli & Simone Albano

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-red)

## Project Overview

The project explores a public cyberbullying dataset and compares several supervised-learning approaches for multiclass tweet classification.

The workflow includes:

- exploratory analysis and class-specific WordClouds;
- NMF exploratory topic analysis;
- TF-IDF with unigrams and bigrams;
- custom stopword handling;
- SMOTETomek resampling;
- KNN with Euclidean and cosine distance;
- Decision Tree tuning with cross-validation;
- Random Forest comparison;
- confusion matrices and per-class evaluation metrics.

The Decision Tree depth is selected using `GridSearchCV` on the training data, so the test split is reserved for final evaluation.

## Repository structure

```text
Cyberbullying-Tweet-Classification/
├── cyberbullying_classification.ipynb
├── requirements.txt
├── README.md
├── README.it.md
├── .gitignore
└── data/
    └── .gitkeep
```

The dataset itself is intentionally excluded from Git.

## Dataset

Public Kaggle dataset:

**Cyberbullying Classification — andrewmvd/cyberbullying-classification**

Expected CSV filename:

```text
cyberbullying_tweets.csv
```

The notebook tries to load the dataset in this order:

1. `data/cyberbullying_tweets.csv`
2. automatic Kaggle download using `kagglehub`
3. manual upload when running in Google Colab

This makes the same notebook usable both locally and in Colab.

## Local installation

Clone the repository:

```bash
git clone https://github.com/zSpiDa/Cyberbullying-Tweet-Classification.git
cd Cyberbullying-Tweet-Classification
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then open the notebook with Jupyter:

```bash
jupyter notebook cyberbullying_classification.ipynb
```

or open the repository in **VS Code**, select the `.venv` Python interpreter/kernel, and run the notebook normally.

## Using the project in Google Colab

Upload or open `cyberbullying_classification.ipynb` in Google Colab.

If some dependencies are missing in the current runtime, run:

```python
%pip install -r requirements.txt
```

when the repository has been cloned inside Colab, or install the few required packages directly:

```python
%pip install imbalanced-learn wordcloud kagglehub
```

Then use **Runtime → Run all**.

The notebook does not depend on a personal Google Drive path.

## Using a local copy of the dataset

If you prefer not to download the dataset automatically, create:

```text
data/cyberbullying_tweets.csv
```

The `data/` directory is ignored by Git so the dataset will not be committed accidentally.

## Reproducibility

The project uses a fixed random state:

```python
RANDOM_STATE = 67
```

The TF-IDF vectorizer is fitted only on the training split and then applied to the test split.

Decision Tree hyperparameter selection is performed through cross-validation on the training data rather than by choosing a depth directly from test-set results.

## Academic context

This project was created for the **Intelligent Systems** course at the University of Bari.

**Authors:** Daniele Spinelli and Simone Albano
