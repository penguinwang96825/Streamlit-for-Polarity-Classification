import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Packages for NLP
import ktrain

# Packages for data preprocessing
import pandas as pd
import numpy as np
import os
import random

def main():
    st.title("DL-based NLP with Streamlit")
    st.subheader("Detect toxicity across a diverse range of conversations.")
    st.markdown('''
    Only using 1 percent of [dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) 
    to train the BERT model.
    ''')

    predictor = ktrain.load_predictor('toxicity_classier')

    def seed_everything(seed = 17):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
    seed_everything()

    def analyze_toxicity(predictor, text):
        temp = predictor.predict([text])
        toxicity = {
            "Overall Toxicity": temp[0][0][1],
            "Severe Toxicity": temp[0][1][1],
            "Obscene": temp[0][2][1],
            "Threat": temp[0][3][1],
            "Insult": temp[0][4][1],
            "Identity Attack": temp[0][5][1],
            "Sexual Explicit": temp[0][6][1]
        }
        data = pd.DataFrame.from_dict([toxicity]).T
        data.columns = ["Probability"]
        return data

    message = st.text_input(label = "Enter your comment", value = "Type here...")
    if st.button("Analyze"):
        data = analyze_toxicity(predictor, message)
        st.dataframe(data.style.highlight_max(axis = 0, color = 'darkorange'))
        st.bar_chart(data)

if __name__ == "__main__":
    main()