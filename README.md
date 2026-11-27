# Sentiment Analysis with TF-IDF

This project implements tweet sentiment classification with **TF-IDF features**
and **Logistic Regression**. The complete experiment is in
[`TF-IDF.ipynb`](TF-IDF.ipynb), with the written analysis in
[`TF-IDF_analysys.pdf`](TF-IDF_analysys.pdf).

## What the project does

The notebook:

1. Loads the train, validation, and test CSV files.
2. Explores the text with word clouds, token-frequency charts, and text-length
   plots.
3. Normalizes tweet text, including contractions, abbreviations, emoticons,
   links, mentions, numbers, and repeated characters.
4. Converts text into TF-IDF features using word n-grams.
5. Trains a Logistic Regression classifier.
6. Evaluates validation predictions with accuracy, precision, recall, F1 score,
   ROC-AUC, and a confusion matrix.
7. Creates a `submission.csv` file with test predictions.

## Model configuration

| Component | Configuration |
| --- | --- |
| Feature extraction | TF-IDF |
| N-gram range | Unigrams through trigrams |
| Maximum features | 300,000 |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.9 |
| Sublinear term frequency | Enabled |
| Logistic Regression solver | `saga` |
| Regularization | L2 (`C=2.0`) |
| Maximum iterations | 1,000 |

## Dataset format

The notebook expects the Kaggle dataset at:

```text
/kaggle/input/ai-2-deep-learning-for-nlp-homework-1/
```

with these files:

```text
train_dataset.csv
val_dataset.csv
test_dataset.csv
```

The training and validation files must contain `Text` and `Label` columns. The
test file must contain `ID` and `Text`. Labels are expected to be binary values
(`0` and `1`).

## Running the notebook

### Kaggle

1. Open `TF-IDF.ipynb` in a Kaggle Notebook.
2. Attach the dataset at the path shown above.
3. Run all cells from top to bottom.
4. Find the generated predictions in `submission.csv`.

### Local Jupyter installation

Install the main Python dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud emoji jupyter
```

Download the NLTK data used by the notebook:

```python
import nltk

nltk.download("punkt")
nltk.download("names")
nltk.download("stopwords")
```

The notebook uses Kaggle-specific absolute paths. Update the three
`pd.read_csv(...)` paths in the loading cell to use your local dataset directory
before running it outside Kaggle.

## Output

The notebook writes `submission.csv` with these columns:

```text
ID, Label
```

## Reproducibility

The notebook chooses a new random state at startup, so the classifier's
`random_state` can differ between runs.

## Repository structure

```text
.
├── TF-IDF.ipynb          # EDA, preprocessing, training, evaluation, inference
├── TF-IDF_analysys.pdf   # Project analysis and discussion
└── README.md             # Project documentation
```
