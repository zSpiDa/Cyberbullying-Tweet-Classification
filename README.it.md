# 🛡️ Cyberbullying Tweet Classification

[🇬🇧 English version](README.md)

**Progetto di Machine Learning e NLP sviluppato per il corso di Sistemi Intelligenti dell'Università degli Studi di Bari (2026)**

**Autori:** Daniele Spinelli & Simone Albano

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-red)

## Panoramica del progetto

Questo progetto affronta la classificazione multiclasse del cyberbullismo su dati provenienti da Twitter utilizzando tecniche di Natural Language Processing e Machine Learning.

Il workflow combina analisi esplorativa del testo, estrazione delle feature tramite TF-IDF, resampling, confronto tra modelli e interpretabilità. L'obiettivo è distinguere sei categorie di cyberbullismo e valutare il comportamento di diversi classificatori su rappresentazioni testuali sparse.

## Dataset e analisi esplorativa

Il progetto utilizza il dataset pubblico **Cyberbullying Classification** di `andrewmvd`, disponibile su Kaggle.

Il dataset utilizzato contiene **47.692 tweet** distribuiti in sei classi relativamente bilanciate:

- age
- ethnicity
- gender
- not_cyberbullying
- other_cyberbullying
- religion

L'analisi esplorativa comprende WordCloud specifiche per classe e un'analisi dei topic basata su NMF.

L'EDA evidenzia pattern lessicali più definiti in alcune categorie, mentre classi più generiche come `other_cyberbullying` presentano un linguaggio più eterogeneo.

## Pipeline di Machine Learning

### 1. Train/Test Split

Il dataset viene suddiviso in training set e test set tramite una divisione stratificata 80/20 con un random state fisso.

### 2. Estrazione delle feature con TF-IDF

I tweet vengono rappresentati tramite `TfidfVectorizer` utilizzando:

- unigrammi e bigrammi;
- un massimo di 10.000 feature;
- stopword inglesi personalizzate;
- `max_df=0.95`;
- `min_df=2`.

Il vectorizer viene addestrato **esclusivamente sui dati di training** e successivamente applicato al test set.

### 3. SMOTETomek

SMOTETomek viene utilizzato come passaggio combinato di resampling e pulizia dei confini decisionali prima dell'addestramento dei modelli KNN e Random Forest.

Poiché il dataset originale è già relativamente bilanciato, questo passaggio non deve essere interpretato semplicemente come bilanciamento delle classi.

SMOTETomek combina oversampling sintetico e pulizia tramite Tomek Links; di conseguenza, la variazione netta del numero di campioni non corrisponde direttamente al numero di osservazioni rimosse.

### 4. Modelli di classificazione

I modelli valutati sono:

- K-Nearest Neighbors con distanza euclidea;
- K-Nearest Neighbors con distanza coseno;
- Decision Tree;
- Random Forest con 5 alberi;
- Random Forest con 200 alberi.

Il parametro `max_depth` del Decision Tree viene selezionato tramite `GridSearchCV`, utilizzando una cross-validation a 5 fold e Macro F1 sui dati di training.

La ricerca ha selezionato:

```text
max_depth = None
CV Macro F1 = 0.8013
```

Il test set rimane separato da questa fase di selezione degli iperparametri e viene utilizzato per la valutazione finale.

## Risultati

| Modello | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| KNN — Euclidean | 0.3659 | 0.6617 | 0.3669 | 0.3775 |
| KNN — Cosine | 0.6960 | 0.7287 | 0.6953 | 0.7083 |
| Decision Tree — Tuned | 0.8032 | 0.8030 | 0.8022 | 0.8026 |
| Random Forest — 5 Trees | 0.8182 | 0.8162 | 0.8173 | 0.8163 |
| **Random Forest — 200 Trees** | **0.8334** | **0.8334** | **0.8327** | **0.8317** |

La **Random Forest con 200 alberi** ha ottenuto le migliori prestazioni complessive tra gli approcci valutati, raggiungendo circa **83,3% di accuracy** e **0,832 di Macro F1**.

Il notebook include inoltre confusion matrix e visualizzazioni di precision, recall e F1-score per classe, consentendo un confronto più dettagliato tra i modelli.

## Interpretabilità

La Gini feature importance della Random Forest viene utilizzata per analizzare le feature TF-IDF più influenti.

Sia singoli termini sia bigrammi contribuiscono al modello, offrendo una semplice interpretazione dei pattern testuali associati al task di classificazione.

## Struttura del repository

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

Il dataset è intenzionalmente escluso da Git.

## Installazione

Clona il repository:

```bash
git clone https://github.com/zSpiDa/Cyberbullying-Tweet-Classification.git
cd Cyberbullying-Tweet-Classification
```

Crea e attiva un ambiente virtuale.

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

Installa le dipendenze:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quindi apri il notebook:

```bash
jupyter notebook cyberbullying_classification.ipynb
```

Il notebook può essere aperto anche direttamente in **VS Code** tramite l'estensione Jupyter.

## Caricamento del dataset

Il notebook è portabile tra ambienti Jupyter locali, VS Code e Google Colab.

Prova a ottenere `cyberbullying_tweets.csv` nel seguente ordine:

1. file locale `data/cyberbullying_tweets.csv`;
2. download automatico di `andrewmvd/cyberbullying-classification` tramite `kagglehub`;
3. caricamento manuale del file quando viene eseguito su Google Colab, se i metodi precedenti non sono disponibili.

Non è necessario alcun percorso personale di Google Drive.

## Google Colab

Apri `cyberbullying_classification.ipynb` in Colab ed esegui il notebook dall'inizio alla fine.

Se nel runtime non sono già presenti i pacchetti aggiuntivi richiesti, installali con:

```python
%pip install imbalanced-learn wordcloud kagglehub
```

Quindi utilizza **Runtime → Run all**.

## Riproducibilità

Il progetto utilizza:

```python
RANDOM_STATE = 67
```

Il repository include il notebook eseguito con gli output della valutazione, in modo che i risultati riportati possano essere verificati direttamente.

## Contesto accademico

Sviluppato per il corso di **Sistemi Intelligenti** dell'**Università degli Studi di Bari** nel 2026.
