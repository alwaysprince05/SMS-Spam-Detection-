import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# making the object of the PorterStemmer as ps
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum(): # used to remove the special character
            y.append(i)

    text = y[:] # copying of the list because we need the clone the list because list is the mutable datatype.
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # now we are doing the stemming inside this
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
if st.button("Predict"):

    # now we are dividing the model into the four steps
    # 1. Preprocess
    # we are sending all the input sms into the transformed text
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    # above vector_input put inside the model.predict and this give 0 or 1 & we put the 0th item inside the result
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
       st.header("🚨 Spam")
    else :
       st.header("✅ Not Spam")



