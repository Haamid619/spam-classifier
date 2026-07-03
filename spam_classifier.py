import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the columns we need
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())
print(df['label'].value_counts())

# Convert spam/ham labels to 1/0
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # ham=0, spam=1

# Convert message text into numbers
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['message'])
y = df['label']

print("Data shape:", X.shape)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape[0])
print("Testing size:", X_test.shape[0])

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained!")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Test the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Test on your own messages
def predict_spam(message):
    msg_tfidf = tfidf.transform([message])
    result = model.predict(msg_tfidf)[0]
    return "SPAM 🚨" if result == 1 else "HAM ✅"

print(predict_spam("To avail an iPhone click here!"))
print(predict_spam("I'm too busy bro, can suzi go instead?"))