# Spam Classifier

A machine learning model that classifies SMS/email messages as spam or not spam with 97.4% accuracy.

## What it does
- Takes a text message as input
- Returns whether it is SPAM or HAM (normal)

## Tech Stack
- Python
- scikit-learn (Naive Bayes, TF-IDF)
- pandas, matplotlib, seaborn

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 97.4% |
| Spam Precision | 100% |
| Spam Recall | 81% |

## Dataset
SMS Spam Collection Dataset from Kaggle — 5572 messages labeled spam or ham.

## How to run
1. Clone the repo
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`
3. Run: `python spam_classifier.py`
