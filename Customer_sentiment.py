import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st

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
    elif predicted_class == 1:
        return 'Neutral'
    else:
        return 'Positive'

def main():
    st.title("Product Review Sentiment Analyzer")

    # User input for product review
    review = st.text_area("Enter your product review:")

    # Analyze sentiment when the user submits a review
    if st.button("Analyze Sentiment"):
        if review:
            sentiment = analyze_sentiment_with_bert(review)
            st.write(f'This is a "{sentiment}" review from the customer.')
        else:
            st.warning("Please enter a product review.")

if __name__ == "__main__":
    main()
