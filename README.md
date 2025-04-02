# 📌 Amazon Review Polarity Classification

This project implements a text classification model to classify Amazon reviews into two categories: positive and negative. The model uses the Amazon Reviews Polarity Dataset, which contains reviews with a rating of 1 or 2 as negative, and 4 or 5 as positive. A Multinomial Naive Bayes classifier is used to classify the reviews based on their text content.

## 🗂️  Dataset

The dataset used for training and testing the model consists of Amazon reviews with the following columns:

- **label**: The class label for the review (1 for negative, 2 for positive).
- **title**: The title of the review.
- **text**: The full text of the review.

The dataset is split into two parts:
- **Training data**: 1.8 million samples for training.
- **Test data**: 200,000 samples for testing.

**Note**: Reviews with a score of 3 are excluded from the dataset, as only ratings of 1, 2, 4, and 5 are used to define negative and positive reviews.

## Objective

The goal of this project is to:
- Preprocess the dataset by cleaning and converting the labels.
- Extract features from the text using TF-IDF (Term Frequency-Inverse Document Frequency).
- Train a Naive Bayes classifier to predict the sentiment of reviews as positive or negative.
- Evaluate the model’s performance using accuracy, classification report, and confusion matrix.

## 🚀 Steps

1. **Data Preprocessing**:
   - Load the training and test datasets.
   - Rename columns to match the expected format (`label`, `title`, `text`).
   - Convert labels from `1` and `2` to `0` (negative) and `1` (positive).
   
2. **Text Vectorization**:
   - Use `TfidfVectorizer` to convert text data into numerical format, which can be used by the model.
   
3. **Model Training**:
   - Train a **Multinomial Naive Bayes** classifier on the training data (after text vectorization).

4. **Prediction**:
   - Predict the sentiment of the test data using the trained model.

5. **Evaluation**:
   - Evaluate the model using various metrics such as accuracy, classification report, and confusion matrix.

## 🔧 Requirements

To run this notebook, you need to install the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
