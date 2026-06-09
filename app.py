import streamlit as st
import tensorflow as tf
import pickle
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model = tf.keras.models.load_model(
    "sentiment_model.keras"
)

tokenizer = pickle.load(
    open("tokenizer.pkl","rb")
)

MAX_LEN=80

stop_words=set(
    stopwords.words('english')
)

lemmatizer=WordNetLemmatizer()

def preprocess_text(text):

    text=text.lower()

    text=re.sub(
        r"http\S+",
        "",
        text
    )

    text=re.sub(
        r"@\w+",
        "",
        text
    )

    text=re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens=word_tokenize(text)

    cleaned=[

        lemmatizer.lemmatize(word)

        for word in tokens

        if word not in stop_words

    ]

    return " ".join(cleaned)

st.title(
    "Sentiment Analysis on Social Media Posts"
)

st.write(
    "Enter a social media post below"
)

user_input=st.text_area(
    "Enter text"
)

if st.button(
    "Predict Sentiment"
):

    cleaned=preprocess_text(
        user_input
    )

    seq=tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded=pad_sequences(
        seq,
        maxlen=MAX_LEN
    )

    prediction=model.predict(
        padded
    )[0][0]

    if prediction>0.5:

        st.success(
            "Positive Sentiment 😀"
        )

        st.write(
            "Confidence:",
            round(prediction*100,2),
            "%"
        )

    else:

        st.error(
            "Negative Sentiment 😞"
        )

        st.write(
            "Confidence:",
            round((1-prediction)*100,2),
            "%"
        )