import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


fake_news_df = pd.read_csv('Fake.csv')  
real_news_df = pd.read_csv('True.csv')  
fake_news_df['label'] = 0
real_news_df['label'] = 1
df = pd.concat([fake_news_df, real_news_df])
df = df[['text', 'label']]
df.dropna(inplace=True)


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('tfidf.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)
