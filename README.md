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
5. **Evaluation:** micro-F1 and macro-F1 on a held-out test split, per-category F1, a comparison with a logistic-regression baseline on the BoW features, and a decision threshold selected on validation data for both models.

A reproducible 70/15/15 train/validation/test split is created before fitting preprocessing. The test split is reserved for the final evaluation. The notebook checks the dataset dimensions and label counts and draws the architecture with Matplotlib.

**Tools:** Python, TensorFlow/Keras (functional API), scikit-learn, NumPy, SciPy, Matplotlib.

## Results

Held-out test set (1,650 news items, 18 categories). Hyperparameters and the decision threshold are chosen on the validation set only.

| Model | Threshold | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- |
| Most frequent categories (reference) | – | 0.268 | 0.062 |
| Logistic regression on BoW | 0.35 | 0.648 | 0.622 |
| **Multi-input neural network** | **0.20** | **0.675** | **0.647** |

- With the default 0.5 threshold the network looks *worse* than the linear baseline (macro-F1 0.52 vs 0.62): it is very precise but misses many labels, because only about 13% of label decisions are positive. Choosing the threshold on validation data fixes this and makes the network the best model.
- The validation loss reaches its minimum after three epochs and then rises, so the network overfits quickly on 7,700 training items; early stopping restores the best weights.
- Per-category F1 ranges from about 0.3 to 0.8, so some categories are much harder than others. The notebook includes the full per-category table and chart.

TensorFlow training on CPU is not bit-for-bit deterministic, so the network's scores can vary slightly between runs.

## Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook DeepLearning_project.ipynb
```

Run the cells from top to bottom (it also works in Google Colab). The data cell looks for `data/input_data.pkl` or `data/input_data.zip` and otherwise tries to download the archive from [the course website](https://frasca.di.unimi.it/MLDNN/input_data.zip); see [Data availability](#data-availability). The dataset is not stored in this repository, and the archive is loaded with Python `pickle`, so only use the file from this source.

## Data availability

The news dataset was provided by the course *Machine Learning, Artificial Neural Networks and Deep Learning* (University of Milan) and is not redistributed here. The download URL used by the notebook now returns a different archive, so to run the notebook place the original `input_data.pkl` (or `input_data.zip`) in a `data/` folder next to it. The data-loading cell checks the archive and stops with an explicit message if it does not contain the expected fields.

## Scope

This project is a baseline for exploring multi-label classification. Its metrics describe one fixed split and a limited hyperparameter search; they do not guarantee performance on other data or splits.
