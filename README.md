📖 Next Word Prediction using LSTM

This project demonstrates an end-to-end implementation of Next Word Prediction using LSTM (Long Short-Term Memory) networks. The model is trained on a text corpus, tested for accuracy, and deployed as an interactive web application using Streamlit.

🚀 Project Workflow

Model Training – experiments.ipynb

      Load and preprocess dataset
      Tokenize text sequences
      Apply padding for uniform input length
      Train an LSTM-based language model
      Save trained model (next_word_pred.h5) and tokenizer (tokenizer.pkl)

Model Testing – prediction.ipynb

      Load the trained model and tokenizer
      Provide test sentences
      Predict the most likely next word
      Evaluate model performance

Deployment – app.py

      Interactive Streamlit app
      User inputs a sequence of words
      Model predicts the next word in real time
      Simple and user-friendly web interface

      <img width="1487" height="856" alt="image" src="https://github.com/user-attachments/assets/450a1b27-83b8-469a-a7e0-d06563420620" />
