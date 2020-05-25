# Streamlit for Toxicity Classification
This is a demo illustrating a classification model, and I trained the simple BERT model ("bert-base-uncased") on a [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) dataset. I did not train the BERT model with the whole dataset, instead, I used 1 persent only as this is not my main goal anyway. To put it succinctly, this BERT model can detect toxicity across a diverse range of conversations.

## Step 1: Download repository
Download repository.

1. `git clone git@github.com:penguinwang96825/Streamlit_for_Toxicity_Classification.git`
2. `cd Streamlit_for_Toxicity_Classification`

## Step 2: Download Pre-trained Model
Download [file](https://drive.google.com/file/d/1i79tQKYwzj_RZIrr0h34vRYSKJRl0p4L/view?usp=sharing) and put it into `Streamlit_for_Toxicity_Classification` folder.

## Step 3: Set up the environment
Download the requirements.txt file and run it in the terminal.

`pip install -r requirements.txt`

## Step 4: Run the app
1. Run the demo in terminal: `streamlit run app.py`
2. View the Streamlit app in your browser: `http://localhost:8501`

![demo]()

## © Copyright
See [License](https://github.com/penguinwang96825/Streamlit-for-Toxicity-Classification/blob/master/LICENSE) for more details.
