# Iris Species Classification

## Video

https://drive.google.com/file/d/1E9C8OM5eoIk8nbt4LRSxRQxtQTEgSdQf/view?usp=drivesdk

## Demo

https://iris-species-classifier-mmnyjpcbydgyybgeldth7i.streamlit.app/

## Academic Information

**Course:** Data Mining  
**University:** Universidad de la Costa  
**Professor:** José Escorcia-Gutierrez, Ph.D.

## Team Members

- Diego Navarro Gómez (Group: 18690)
- Juan Félix (Group: 18038)
- Dinellys García (Group: 18038)
- Kimberly Ochoa (Group: 19027)

## Objective

Train a classification model capable of predicting the species of an Iris flower based on its sepal and petal measurements, and communicate the results through an interactive dashboard.

## Methodology

1. **Data Understanding** — Iris dataset (150 samples, 3 balanced classes, 4 numerical features, no missing values).
2. **Preprocessing** — Feature normalization using StandardScaler.
3. **Train/Test Split** — 75% training / 25% test, stratified.
4. **Modeling** — Random Forest Classifier (100 trees). Selected for its robustness, low tuning requirements, and interpretability via feature importances.
5. **Evaluation** — Accuracy, Precision, Recall, and F1-Score on the held-out test set.

## Dashboard

The dashboard includes:

- Model metrics: Accuracy, Precision, Recall, and F1-Score.
- Prediction panel with sliders to enter measurements and a 3D scatter plot showing the new sample vs the dataset.
- Data exploration: histograms, scatter plot, and scatter matrix.

## Installation

```bash
pip install -r requirements.txt
streamlit run Proyect.py
```
