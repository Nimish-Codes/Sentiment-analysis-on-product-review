import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd

# Example labeled training data
train_data = pd.DataFrame({
    'review': ["This is a positive review.", "bloody shit", "Neutral comment."],
    'label': [2, 0, 1]  # 0: Negative, 1: Neutral, 2: Positive
})


def fine_tune_bert_for_sentiment(train_data, num_epochs=3):
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Three classes: Negative, Neutral, Positive

    # Tokenize and encode the training data
    inputs = tokenizer(
        train_data['review'],
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    labels = torch.tensor(train_data['label'])

    # Create a DataLoader for training data
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, label = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    return model

def analyze_sentiment_with_fine_tuned_bert(model, review):
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

# Streamlit App
st.title('Sentiment Analysis with Fine-Tuned BERT')
st.write('This app performs sentiment analysis on user-provided text using a fine-tuned BERT model.')

# Input field for user
user_review = st.text_area('Enter your review:')
if st.button('Analyze Sentiment'):
    if user_review:
        fine_tuned_model = fine_tune_bert_for_sentiment(train_data)  # Replace train_data with your labeled training dataset
        sentiment = analyze_sentiment_with_fine_tuned_bert(fine_tuned_model, user_review)
        st.write(f'This is a "{sentiment}" review from the user.')
    else:
        st.warning('Please enter a review.')
