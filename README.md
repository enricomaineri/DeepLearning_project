# DeepLearning_project

A multi-input deep neural network for **multi-label news classification**, implemented in a Jupyter notebook for the *Machine Learning, Artificial Neural Networks and Deep Learning — Part 2* course (exam session: 16 June 2026).

The repository is currently private while the project is being reviewed.

## Task

The supplied course dataset contains 11,000 news entries and 18 possible semantic categories. Each article can have multiple labels (two or three per item). Inputs are:

- the article headline and short description as variable-length text;
- a 10,000-feature Bag-of-Words count vector;
- publication year (2012–2022) and month.

The target is a binary multi-hot matrix, one column per category.

## Model and pipeline

The notebook follows the assignment's five sections:

1. **Model choice:** a three-branch neural network for the text, BoW, and metadata inputs.
2. **Input preparation:** text cleaning, a vocabulary fitted on training documents only, integer encoding and padding, training-only rare-term filtering and BoW standardization, and training-only one-hot encoding of year and month.
3. **Output and loss:** 18 independent sigmoid outputs trained with binary cross-entropy, appropriate for non-exclusive labels.
4. **Model configuration:** Embedding + LSTM for text; Dense + Batch Normalization + ReLU for BoW; metadata concatenation; a dense classification head with dropout, L2 regularization, Adam, gradient clipping, and early stopping. A small search compares configurations using validation macro-F1.
5. **Evaluation:** micro-F1 and macro-F1 on a held-out test split at a fixed 0.5 threshold.

A reproducible 70/15/15 train/validation/test split is created before fitting preprocessing. The test split is reserved for the final evaluation. The notebook also checks the expected dataset dimensions and label counts and draws the architecture with Matplotlib.

## Run the notebook

1. Open `DeepLearning_project.ipynb` in Google Colab or a Jupyter environment with TensorFlow, scikit-learn, SciPy, and Matplotlib installed.
2. Run the cells from top to bottom.
3. The setup cell downloads the course archive from [the course data URL](https://frasca.di.unimi.it/MLDNN/input_data.zip). The dataset itself is not stored in this repository.

The notebook's saved outputs have been cleared so the results are generated in the environment where it is run. Use only the dataset archive provided by the course, since the notebook loads the downloaded archive with Python pickle.

## Scope

This is an educational baseline, not a production system. Its metrics refer to one fixed split and a limited hyperparameter search; they do not guarantee performance on other data or splits.
