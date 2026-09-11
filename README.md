# 🛡️ Cyberbullying Tweet Classification

**Machine Learning & NLP project developed for the Intelligent Systems course at the University of Bari (2026)**

**Authors:** Daniele Spinelli & Simone Albano

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-red)

## Project Overview

This project investigates multiclass cyberbullying classification on Twitter data using Natural Language Processing and traditional Machine Learning techniques.

The workflow combines exploratory text analysis, TF-IDF feature extraction, resampling, model comparison and interpretability. The goal is to distinguish between six cyberbullying categories and evaluate how different classifiers behave on sparse textual features.

## Dataset & Exploratory Data Analysis

The project uses the public **Cyberbullying Classification** dataset by `andrewmvd` on Kaggle.

The executed dataset contains **47,692 tweets** across six relatively balanced classes:

- age
- ethnicity
- gender
- not_cyberbullying
- other_cyberbullying
- religion

Exploratory analysis includes class-specific WordClouds and an NMF-based topic analysis. The EDA highlights clearer lexical patterns in some categories, while broader classes such as `other_cyberbullying` contain more heterogeneous language.

## Machine Learning Pipeline

### 1. Train/Test Split

The dataset is split into training and test sets using an 80/20 stratified split with a fixed random state.

### 2. TF-IDF Feature Extraction

Tweets are represented using `TfidfVectorizer` with:

- unigrams and bigrams;
- a maximum of 10,000 features;
- custom English stopwords;
- `max_df=0.95`;
- `min_df=2`.

The vectorizer is fitted **only on the training data** and then applied to the test set.

### 3. SMOTETomek

SMOTETomek is used as a combined resampling and boundary-cleaning step before training KNN and Random Forest models.

Because the original dataset is already relatively balanced, this step should not be interpreted simply as class balancing. SMOTETomek combines synthetic oversampling with Tomek Links cleaning, so the net change in sample count does not correspond directly to a number of removed observations.

### 4. Classification Models

The evaluated models are:

- K-Nearest Neighbors with Euclidean distance;
- K-Nearest Neighbors with cosine distance;
- Decision Tree;
- Random Forest with 5 trees;
- Random Forest with 200 trees.

Decision Tree `max_depth` is selected with `GridSearchCV` using 5-fold cross-validation and Macro F1 on the training data. The search selected:

```text
max_depth = None
CV Macro F1 = 0.8013
```

The test set is kept separate from this hyperparameter-selection step and is used for final evaluation.

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| KNN — Euclidean | 0.3659 | 0.6617 | 0.3669 | 0.3775 |
| KNN — Cosine | 0.6960 | 0.7287 | 0.6953 | 0.7083 |
| Decision Tree — Tuned | 0.8032 | 0.8030 | 0.8022 | 0.8026 |
| Random Forest — 5 Trees | 0.8182 | 0.8162 | 0.8173 | 0.8163 |
| **Random Forest — 200 Trees** | **0.8334** | **0.8334** | **0.8327** | **0.8317** |

The **Random Forest with 200 trees** achieved the strongest overall performance among the evaluated approaches, reaching approximately **83.3% accuracy** and **0.832 Macro F1**.

The notebook also includes confusion matrices and class-level precision, recall and F1-score visualizations for a more detailed comparison.

## Interpretability

Random Forest Gini feature importance is used to inspect influential TF-IDF features. Both individual terms and bigrams contribute to the model, providing a simple view of the textual patterns associated with the classification task.

## Repository Structure

```text
Cyberbullying-Tweet-Classification/
├── cyberbullying_classification.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── .gitkeep
```

The dataset itself is intentionally excluded from Git.

## Installation

Clone the repository:

```bash
git clone https://github.com/zSpiDa/Cyberbullying-Tweet-Classification.git
cd Cyberbullying-Tweet-Classification
```

Create and activate a virtual environment.

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then open:

```bash
jupyter notebook cyberbullying_classification.ipynb
```

The notebook can also be opened directly in **VS Code** with the Jupyter extension.

## Dataset Loading

The notebook is portable across local Jupyter environments, VS Code and Google Colab.

It tries to obtain `cyberbullying_tweets.csv` in this order:

1. local file at `data/cyberbullying_tweets.csv`;
2. automatic download of `andrewmvd/cyberbullying-classification` through `kagglehub`;
3. manual file upload when running in Google Colab, if the previous methods are unavailable.

No personal Google Drive path is required.

## Google Colab

Open `cyberbullying_classification.ipynb` in Colab and run the notebook from top to bottom.

If the runtime does not already contain the required additional packages, install them with:

```python
%pip install imbalanced-learn wordcloud kagglehub
```

Then use **Runtime → Run all**.

## Reproducibility

The project uses:

```python
RANDOM_STATE = 67
```

The final repository includes the executed notebook with its evaluation outputs so that the reported results can be inspected directly.

## Academic Context

Developed for the **Intelligent Systems** course at the **University of Bari** in 2026.
