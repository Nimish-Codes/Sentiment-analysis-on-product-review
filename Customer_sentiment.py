import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

def analyze_sentiment_with_bert(review):
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Three classes: Negative, Neutral, Positive

    # Tokenize and encode the review
    encoded_review = tokenizer.encode_plus(
        review,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Get the model predictions
    with torch.no_grad():
        outputs = model(**encoded_review)

    # Convert logits to probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    # Determine the predicted sentiment
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Class 0: Negative, Class 1: Neutral, Class 2: Positive
    if predicted_class == 0:
        return 'Negative'
    elif predicted_class == 0.5:
        return 'Neutral'
    else:
        return 'Positive'

# Streamlit App
st.title('Sentiment Analysis with BERT')
st.write('This app performs sentiment analysis on user-provided text using a pre-trained BERT model.')

# Input field for user
user_review = st.text_area('Enter your review:')
if st.button('Analyze Sentiment'):
    if user_review:
        sentiment = analyze_sentiment_with_bert(user_review)
        st.write(f'This is a "{sentiment}" review from the user.')
    else:
        st.warning('Please enter a review.')
