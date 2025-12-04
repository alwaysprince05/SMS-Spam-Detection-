import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# Print attributes to confirm training
print("class_count_:", model.class_count_)
print("feature_count_ shape:", model.feature_count_.shape)

# Save trained files
pickle.dump(cv, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("Training Completed. Files Saved Successfully.")