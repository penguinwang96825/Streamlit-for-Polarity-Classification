import streamlit as st
import re
import emoji
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Packages for NLP
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# Packages for data preprocessing
import pandas as pd
import numpy as np
import os
import random


def seed_everything(seed=17):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def give_emoji_free_text(text):
    allchars = [string for string in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    cleaned_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return cleaned_text

def clean_text(text):
    # Remove emoji
    text = give_emoji_free_text(text)
    # Remove punctuation
    text = re.sub(r'[^\w\d\s]+', '', text)
    # Remove digits
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def argmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)

class KKBOXSentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(self.bert_layer.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert_layer(
            input_ids=input_ids, 
            attention_mask=attention_mask)
        main = self.dropout(pooled_output)
        return F.softmax(self.classifier(main), dim=1)

def analyze_polarity(text, model, tokenizer, class_name):
    text = clean_text(text)
    encoding = tokenizer.encode_plus(
        text, 
        max_length=128, 
        add_special_tokens=True, 
        return_token_type_ids=False, 
        pad_to_max_length=True, 
        return_attention_mask=True, 
        return_tensors="pt")
    
    logits = model(encoding["input_ids"].to("cpu"), encoding["attention_mask"].to("cpu"))
    logits = logits.to("cpu")
    logits = logits.tolist()[0]
    prediction = class_name[argmax(logits)]
    return logits, prediction

def main():
    seed_everything()
    st.title("Detect Polarity of Japanese Reviews")
    st.subheader("Text classification for reviews of UtaPass & KKBOX using BERT.")
    st.markdown('''
    This sentiment classification task is based on reviews data of UtaPass and KKBOX from Google Play platform. 
    As a KKStreamer at KKBOX, I become more interested in Natural Language Processing, especially 
    text classification.
    ''')

    PRE_TRAINED_MODEL_NAME = 'bert-base-japanese'
    tokenizer = BertJapaneseTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    class_name = ["Negative", "Positive"]
    model = torch.load("best_bert_japanese.pkl")
    model.to("cpu")

    message = st.text_input(label="Enter your comment: ", value="Type here...")
    if st.button("Analyze"):
        message = clean_text(message)
        logits, prediction = analyze_polarity(message, model, tokenizer, class_name)
        pos = "Positive: {}".format(round(logits[1], 4))
        neg = "Negative: {}".format(round(logits[0], 4))
        tokenized_text = tokenizer.tokenize(message)
        st.markdown("**Tokenized text: **")
        st.markdown(" ".join(tokenized_text))
        st.markdown("**Prediction: **")
        st.text(pos)
        st.text(neg)

if __name__ == "__main__":
    main()