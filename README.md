# 🛡️ Cyberbullying Tweet Classification

**Machine Learning & NLP project developed for the Intelligent Systems course at the University of Bari (2026)**  
*Authors: Daniele Spinelli & Simone Albano*

[IT Versione italiana](README.it.md)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-red)

## 📌 Project Overview

This Natural Language Processing (NLP) project focuses on the **multiclass classification of tweets according to different types of cyberbullying**.

The goal is not only to train a predictive model, but also to investigate patterns in online abusive language and understand how textual context, feature extraction and data cleaning affect classification performance.

The project covers the complete Machine Learning workflow, from **Exploratory Data Analysis (EDA)** and text preprocessing to model training and feature importance analysis.

## 📊 Dataset & Exploratory Data Analysis

The project uses a multiclass dataset containing tweets labelled according to different categories of cyberbullying.

During the exploratory analysis, category-specific **WordClouds** were generated to investigate recurring terms and semantic differences between classes.

The analysis highlighted some relevant characteristics of the dataset:

- Categories such as `Religion` and `Age` contain relatively distinctive groups of words.
- `Other_Cyberbullying` and `Not_Cyberbullying` show a higher degree of overlap and noise.
- Some tweets appear to be incorrectly labelled in the original dataset or associated with recurring topics unrelated to explicit cyberbullying, such as the Australian reality show *MKR*.

These observations influenced the preprocessing and resampling strategy adopted in the following stages.

## ⚙️ Machine Learning Pipeline

The classification workflow consists of three main stages.

### 1. Text Preprocessing & Feature Extraction

Tweets are transformed into numerical representations using `TfidfVectorizer`.

The preprocessing pipeline includes:

- Custom stopword filtering and removal of elements such as tags, URLs and `"RT"`.
- **Unigrams and bigrams (`ngram_range=(1,2)`)** to capture contextual information such as `"high school"`.
- Feature space limited to the **10,000 most relevant features**.

### 2. Data Cleaning with SMOTETomek

`SMOTETomek` is applied to improve the quality of the training data.

Since the original dataset is already relatively balanced, the main purpose of this step is not simply class balancing. In particular, the **Tomek Links** component helps remove ambiguous samples located near decision boundaries between overlapping classes.

This is especially relevant for categories where the exploratory analysis revealed substantial semantic overlap.

### 3. Classification Models

Two Machine Learning algorithms are evaluated:

- **K-Nearest Neighbors (KNN)** using cosine distance.
- **Random Forest Classifier**, used as the main classification model.

## 🚀 Results & Interpretability

The **Random Forest Classifier** showed the strongest overall performance among the evaluated approaches.

To better understand the model's predictions, **feature importance based on Gini importance** was analyzed.

The analysis revealed meaningful patterns in the features learned by the model:

- Contextual information captured through bigrams contributes to classification. For example, `"high school"` appears among the most discriminative features.
- Several of the most important textual features are strongly associated with the dataset's cyberbullying categories, including ethnicity, sexual orientation and religion.

This analysis provides an interpretable view of the textual patterns used by the classifier rather than relying exclusively on predictive performance.

## 📂 Repository Structure

- `Tesi_SI_SpinelliAlbano_CyberbullyingTweets.ipynb` — Jupyter Notebook containing the complete workflow, from EDA to feature importance analysis.
- `tesi_si_spinellialbano_cyberbullyingtweets.py` — Python script containing the project implementation.
- `Tesina Cyberbullying Tweets Spinelli Albano Sistemi Intelligenti 2026.pdf` — Final academic report and project presentation.

## 📄 Dataset

The dataset used in this project is publicly available on Kaggle:

**Cyberbullying Classification Dataset — Andrew MVD**  
https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

## 🎓 Academic Context

Developed as a university project for the **Intelligent Systems** course at the **University of Bari**, 2026.

**Authors:** Daniele Spinelli & Simone Albano
