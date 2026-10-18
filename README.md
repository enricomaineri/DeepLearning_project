# DeepLearning_project

A multi-input deep neural network for **multi-label news classification**. The project combines article text, Bag-of-Words (BoW) features, and publication metadata to predict multiple semantic categories for each news item.

## Problem and data

The dataset contains 11,000 news entries and 18 possible categories. Each item can have more than one label. The model uses:

- the article headline and short description as variable-length text;
- a 10,000-feature BoW count vector;
- publication year (2012–2022) and month.

Labels are represented as a binary multi-hot matrix, with one column per category.

## Model and pipeline

The notebook is organized around the complete classification workflow:

1. **Model choice:** a three-branch neural network for text, BoW, and metadata.
2. **Input preparation:** text cleaning, a vocabulary fitted on training documents only, integer encoding and padding, training-only rare-term filtering and BoW standardization, and training-only one-hot encoding of year and month.
3. **Output and loss:** 18 independent sigmoid outputs trained with binary cross-entropy, suitable for non-exclusive categories.
4. **Model configuration:** Embedding + LSTM for text; Dense + Batch Normalization + ReLU for BoW; a dense classification head with dropout and L2 regularization; Adam optimization, gradient clipping, and early stopping. A compact search compares configurations using validation macro-F1.
5. **Evaluation:** micro-F1 and macro-F1 on a held-out test split at a fixed 0.5 threshold, per-category F1, and a comparison with a logistic-regression baseline on the BoW features.

A reproducible 70/15/15 train/validation/test split is created before fitting preprocessing. The test split is reserved for the final evaluation. The notebook checks the dataset dimensions and label counts and draws the architecture with Matplotlib.

**Tools:** Python, TensorFlow/Keras (functional API), scikit-learn, NumPy, SciPy, Matplotlib.

## Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook DeepLearning_project.ipynb
```

Run the cells from top to bottom (it also works in Google Colab). The data cell downloads the archive from [the course website](https://frasca.di.unimi.it/MLDNN/input_data.zip) into a local `data/` folder the first time it runs; if the download fails, download `input_data.zip` manually and place it in `data/`. The dataset is not stored in this repository, and the archive is loaded with Python `pickle`, so only use the file from this source.

## Scope

This project is a baseline for exploring multi-label classification. Its metrics describe one fixed split and a limited hyperparameter search; they do not guarantee performance on other data or splits.
