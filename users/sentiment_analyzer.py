import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, GPT2ForSequenceClassification, GPT2Tokenizer
import torch
import openai
import gdown
import os

#download model from google drive
# gpt2model_dir = './model'
# gpt2_url = 'https://drive.google.com/drive/folders/175Skml7CqvbfgS14oOctFzx-7go72jak?usp=drive_link'
bertmodel_dir = './modelBERT'
bert_url = 'https://drive.google.com/drive/folders/1_mugHOAaCPIUgXX6l9pfLICuF5kwGIiz?usp=drive_link'
if not os.path.exists(bertmodel_dir):
    gdown.download_folder(bert_url)

st.title('🤖 Sentiment Analyzer')

#Sentiment Analyzer
st.subheader('Sentiment Analyzer')
st.markdown('#### Please put your online news headline here: ')
headline_input = st.text_input('headline', label_visibility="collapsed")
st.button('Predict')

# openai.api_key = "YOUR_API_KEY"

# response = openai.ChatCompletion.create(
#     model="fine-tuned-model-id",
#     messages=[
#         {"role": "user", "content": "Your input here"}
#     ]
# )

# print(response['choices'][0]['message']['content'])
#------------------------------------------------------------------------------------------------------------------------------
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
# model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3, output_hidden_states=True)
# Load the tokenizer and model
@st.cache_resource
def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(bertmodel_dir)
    return tokenizer

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(bertmodel_dir)
    return model

# @st.cache_resource
# def load_model():
#     model = GPT2ForSequenceClassification.from_pretrained(gpt2model_dir)
#     return model

# @st.cache_resource
# def load_tokenizer():
#     tokenizer = GPT2Tokenizer.from_pretrained(gpt2model_dir)
#     return tokenizer
#------------------------------------------------------------------------------------------------------------------------------
# Tokenize the input
# inputs = tokenizer(headline_input, return_tensors='pt')
def classify_sentiment(text):
    #Check if the input is empty
    if not text.strip():
        return "Invalid input: Text is empty. Please provide valid input."
    # Encode the text
    tokenizer = load_tokenizer()
    encoded_text = tokenizer.encode(text, return_tensors="pt")
    # Predict the sentiment
    model = load_model()
    sentiment = model(encoded_text)[0]
    # Decode the sentiment
    return sentiment.argmax().item()

# def predict_sentiment(text):
#     '''Function to predict the sentiment of a given text using a pre-trained BERT model.
#     Args: the input text for sentiment prediction.
#     Returns: the predicted sentiment ('negative', 'neutral', 'positive').
#     '''

#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
#     outputs = model(**inputs)
#     predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     predicted_class = torch.argmax(predictions, dim=1).item()
#     sentiment = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}
#     return sentiment[predicted_class]

# label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
# predicted_sentiment = [label_map[label.item()] for label in predicted_sentiment_label]
# print(f"Predicted Sentiment: {predicted_sentiment}")

# # Perform inference
# with torch.no_grad():
#     outputs = model(**inputs)

# # Get the prediction
# logits = outputs.logits
# prediction = torch.argmax(logits, dim=1).item()
predicted_sentiment_label = classify_sentiment(headline_input)
sentiments = {0: 'Neutral', 2: 'Negative', 1: 'Positive'}

#------------------------------------------------------------------------------------------------------------------------------
companies = ['aapl', 'meta', 'msft', 'amzn', 'tsla']
company_keywords = {
    'aapl': ['apple', 'aapl'],
    'meta': ['facebook', 'meta'],
    'msft': ['microsoft', 'msft'],
    'amzn': ['amazon', 'amzn'],
    'tsla': ['tesla', 'tsla']
}
related_company = '-'

if headline_input != '':
    for company, keywords in company_keywords.items():
        if any(keyword in headline_input.lower() for keyword in keywords):
            related_company = company
            break
    st.markdown(f'Related company: {related_company.upper()}')
    st.markdown(f'Sentiment: {sentiments[predicted_sentiment_label]}')
    # st.markdown(f'Predicted Sentiment: {predicted_sentiment}')
