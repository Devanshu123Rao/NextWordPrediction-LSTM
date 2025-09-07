import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

##loading the model and tokenizer
model=load_model('next_word_pred.h5')
with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)


def pred_next_word(model,tokenizer,text,max_sequence_length):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_length:
        token_list=token_list[-(max_sequence_length-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_length-1,padding='pre')
    predicted=model.predict(token_list,verbose=1)
    predicted_word_idx=np.argmax(predicted,axis=1)

    for word , index in tokenizer.word_index.items():
        if index==predicted_word_idx:
            return word
    return None


## Streamlit app
st.title("Next Word Prediction using LSTM")

input_text=st.text_input("Enter the word sequence or sentence")
if st.button("Predicting next word"):
    if not input_text.strip():  # <-- check for empty or spaces only
        st.warning("⚠️ Please enter words before predicting.")
    else:
        max_sequence_length=model.input_shape[1]+1
        next_word=pred_next_word(model,tokenizer,input_text,max_sequence_length=max_sequence_length)
        st.write(f'Next Word :- {next_word}')
    