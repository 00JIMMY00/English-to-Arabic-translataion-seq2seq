import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown

# Function to download and cache the model file
@st.cache_data
def download_model(url, output):
    gdown.download(url, output, quiet=False, fuzzy=True)
    return output

# Function to load the model from the cached file
@st.cache_data(show_spinner=False)
def load_my_model(model_path):
    return load_model(model_path)

# Function to download and cache the tokenizer (pickle file)
@st.cache_data
def download_tokenizer(url, output):
    gdown.download(url, output, quiet=False, fuzzy=True)
    return output

# Function to load the tokenizer from the cached pickle file
@st.cache_data
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        return pickle.load(handle)

# Load the encoder model (downloaded if not cached)
encoder_model_url = "https://drive.google.com/file/d/12g6AJgPoFdqX_kzrCSj4n51TyX3iqCfM/view?usp=sharing"
encoder_path = download_model(encoder_model_url, "encoder.keras")
encoder_model = load_my_model(encoder_path)

# Load the decoder model (downloaded if not cached)
decoder_model_url = "https://drive.google.com/uc?id=1Jfiyzov3PXfXE0_8CanowVQj7ERyreER"
decoder_path = download_model(decoder_model_url, "eng_to_arabic_decoder_98_acc_99_val_v2.keras")
decoder_model = load_my_model(decoder_path)

# Load the arabic tokenizer (downloaded if not cached)
arabic_tokenizer_url = "https://drive.google.com/uc?id=1apVvx6g56eC1Fbk4EoDZiikYl9fAA6dX"
arabic_path = download_tokenizer(arabic_tokenizer_url, "arabic_tokenizer.pickle")
arabic_tokenizer = load_tokenizer(arabic_path)

# Load the english tokenizer (downloaded if not cached)
english_tokenizer_url = "https://drive.google.com/uc?id=14IKD2Kd28RbqqIpTChCSibCw3SaDzY3x"
english_path = download_tokenizer(english_tokenizer_url, "english_tokenizer.pickle")
english_tokenizer = load_tokenizer(english_path)

# Translation function
def translate_sentence(sentence, english_tokenizer, arabic_tokenizer, encoder_model, decoder_model, max_encoding_len=13, max_decoding_len=15):
    # Tokenize the input English sentence using english_tokenizer
    input_seq = english_tokenizer.texts_to_sequences([sentence])
    
    # Pad the tokenized input to max_encoding_len
    input_seq = pad_sequences(input_seq, maxlen=max_encoding_len, padding='post')

    # Encode the input as state vectors using the encoder model
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate an empty target sequence of length 1
    target_seq = np.zeros((1, 1))  # [[0]]

    # Populate the first character of the target sequence with the start token ('<sos>')
    target_seq[0, 0] = arabic_tokenizer.word_index['<sos>']

    # Sampling loop to generate the Arabic sentence
    stop_condition = False
    translated_sentence = ''
    
    while not stop_condition:
        # Predict the next token and hidden states (h, c) from the decoder model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample the token with the highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = arabic_tokenizer.index_word.get(sampled_token_index, '<unk>')

        # Append the sampled word to the translated sentence
        translated_sentence += ' ' + sampled_word

        # Stop if we encounter the end token ('<eos>') or exceed max_decoding_len
        if sampled_word == '<eos>' or len(translated_sentence.split()) > max_decoding_len:
            stop_condition = True
        else:
            # Update the target sequence with the sampled token
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update the states (h, c) for the next prediction
            states_value = [h, c]

    # Return the translated sentence without '<eos>' token
    translated_sentence = translated_sentence.replace('<eos>', '').strip()
    return translated_sentence

# Streamlit app
st.title("English to Arabic Translation")
input_text = st.text_input("Enter your English text", "To be or not to be")

if st.button("Translate"):
    translated_text = translate_sentence(input_text, english_tokenizer, arabic_tokenizer, encoder_model, decoder_model)
    st.write(f'Translated Text: {translated_text}')
# print("ss")
# print(translate_sentence("He will play football tommorow",english_tokenizer, arabic_tokenizer, encoder_model, decoder_model ))
# print("hh")
