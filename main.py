# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model


# word_index = imdb.get_word_index()
# reverse_word_index ={value:key for key, value in word_index.items()}

# model =load_model("simple_rnn_imdb.h5")
# def decode_review(text):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])


# def preprocess_text(text):
#     # Encode the text using the word index
    
#     encoded = [word_index.get(word, 2)+3 for word in text.lower().split()]
#     # Pad the sequence to max length 500
#     padded = sequence.pad_sequences([encoded], maxlen=500, padding='pre')
#     return padded

# ## step predictiopn function

# def predict_sentiment(review):
#     preprocessed_input=preprocess_text(review)
#     prediction =model.predict(preprocessed_input)
#     sentiment ='positive ' if prediction[0][0] >0.5 else "Negative"
#     return sentiment,prediction[0][0]


# ### stream lit app

# import streamlit as st

# st.title('IMDb Movie Review Sentiment Analysis')
# st.write('Enter a movie review to classify it as positive or negative.')

# user_input = st.text_area('Movie Review')
# if st.button('Classify'):
#     processed_input = preprocess_text(user_input)
#     prediction =model.predict(processed_input)
#     sentiment = predict_sentiment(processed_input)
#     st.write(f'Sentiment: {sentiment}')
# else:
#     st.write('Please enter a movie review.')


# st.write('Sample review: This movie was fantastic and so thrilling.')

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index and reverse word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model("simple_rnn_imdb.h5")

# Decode integer sequences back to words (optional utility)
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# Preprocess raw input text
def preprocess_text(text):
    # Encode text using the word index
    encoded = [word_index.get(word, 2) + 3 for word in text.lower().split()]
    # Pad the sequence to max length 500
    padded = sequence.pad_sequences([encoded], maxlen=500, padding='pre')
    return padded

# Predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'
    return sentiment, prediction[0][0]

# Streamlit UI
st.title('🎬 IMDb Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('✍️ Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("Please enter a movie review before clicking Classify.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Confidence:** {confidence:.2f}')
else:
    st.info('Enter a review and click "Classify" to begin.')

# Optional sample
st.write('📝 Sample review: _"This movie was fantastic and so thrilling."_')
