# 🛡️ Cyberbullying Tweet Classification
**Progetto per il corso di Sistemi Intelligenti UniBa(2026)**  
*Autori: Daniele Spinelli & Simone Albano*

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-red)

## 📌 Panoramica del Progetto
Questo progetto di Natural Language Processing (NLP) punta a classificare i tweet in base al tipo di cyberbullismo contenuto. 
L'obiettivo non è solo addestrare un modello predittivo, ma **analizzare semanticamente il linguaggio dell'odio online**, comprendendo come il contesto (n-grams) e la pulizia dei dati (resampling) influenzino le performance degli algoritmi.

## 📊 Il Dataset e l'Analisi Esplorativa (EDA)
Abbiamo utilizzato un dataset multiclasse di tweet etichettati. Tramite la generazione di **WordCloud** filtrate per categoria, abbiamo evidenziato differenze semantiche cruciali:
* Classi come `Religion` e `Age` possiedono cluster di parole molto definiti.
* La classe `Other_Cyberbullying` e `Not_Cyberbullying` presentano forte sovrapposizione e "rumore" (es. tweet etichettati erroneamente alla fonte o legati all'anomalia del reality show australiano *MKR*).

## ⚙️ Pipeline di Machine Learning
Per gestire il rumore di fondo e i confini sfumati tra le classi, abbiamo sviluppato la seguente pipeline:

1. **Text Pre-processing & Feature Extraction:**
   * Utilizzo di `TfidfVectorizer` con stopwords personalizzate (rimozione di tag, URL, "RT").
   * **Integrazione dei Bigrammi (`ngram_range=(1,2)`)** per catturare il contesto semantico (es. *"high school"* invece di parole separate).
   * Feature Selection limitata alle 10.000 feature più rilevanti.

2. **Bilanciamento e Pulizia dei Confini (SMOTETomek):**
   * Per risolvere l'ambiguità evidenziata nell'EDA, abbiamo applicato `SMOTETomek`. Più che per bilanciare (il dataset era già distribuito equamente), la componente **Tomek Links** è stata fondamentale per rimuovere chirurgicamente i campioni rumorosi ai confini decisionali tra le classi.

3. **Modellazione:**
   * K-Nearest Neighbors (KNN) con metrica Coseno.
   * **Random Forest Classifier** (Modello principale).

## 🚀 Risultati Principali e Interpretabilità
Il modello Random Forest ha dimostrato un'ottima capacità di generalizzazione.
Analizzando la **Feature Importance (Gini Decrease)**, abbiamo verificato che il modello ha appreso pattern corretti:
* Il modello riconosce l'importanza del contesto (il bigramma `"high school"` è tra le feature più discriminanti).
* Le top 20 parole estratte dall'algoritmo si mappano perfettamente sulle categorie sociologiche del dataset (etnia, orientamento sessuale, religione).

## 📂 Struttura della Repository
* `Tesi_SI_SpinelliAlbano_CyberbullyingTweets.ipynb`: Notebook Jupyter contenente l'intera pipeline commentata, dall'EDA alla Feature Importance.
* `tesi_si_spinellialbano_cyberbullyingtweets.py`: Script Python del progetto.
* `Tesina Cyberbullying Tweets Spinelli Albano Sistemi Intelligenti 2026.pdf`: Report finale e presentazione del progetto.

## 📄 Dataset
* https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
