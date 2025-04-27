import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load datasets
train_df = pd.read_csv('train.csv', header=None)
test_df = pd.read_csv('test.csv', header=None)

train_df.columns = ['Category', 'Title', 'Description']
test_df.columns = ['Category', 'Title', 'Description']

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

train_df['text'] = train_df['Title'] + " " + train_df['Description']
test_df['text'] = test_df['Title'] + " " + test_df['Description']

train_df['clean_text'] = train_df['text'].apply(preprocess)
test_df['clean_text'] = test_df['text'].apply(preprocess)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

y_train = train_df['Category']
y_test = test_df['Category']

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'news_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
