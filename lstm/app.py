import pickle
import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Add error handling for model loading
@st.cache_resource
def load_lstm_model():
    try:
        # Try loading with compile=False first
        model = load_model('next_word_lstm.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Trying alternative loading method...")
        try:
            # Alternative: Load with custom objects
            model = keras.models.load_model('next_word_lstm.h5', compile=False)
            return model
        except Exception as e2:
            st.error(f"Alternative loading also failed: {str(e2)}")
            return None

@st.cache_resource
def load_tokenizer():
    try:
        if not os.path.exists('tokenizer.pickle'):
            st.error("tokenizer.pickle file not found!")
            return None
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) == 0:
            return "No valid tokens found"
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted, axis=-1)[0]
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                return word
        return 'Unknown'
    except Exception as e:
        return f"Error: {str(e)}"

# Main app
st.title('Next Word Prediction with LSTM and Early Stopping')

# Load model and tokenizer
model = load_lstm_model()
tokenizer = load_tokenizer()

if model is None or tokenizer is None:
    st.error("Failed to load model or tokenizer. Please check your files.")
    st.stop()

input_text = st.text_input('Enter the sequence of words:', 'to be or not to')

if st.button('Predict Next Word'):
    if input_text.strip():
        max_sequence_len = model.input_shape[1] + 1  # Add 1 since we pad to maxlen-1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f'**Predicted next word:** {next_word}')
    else:
        st.warning("Please enter some text!")