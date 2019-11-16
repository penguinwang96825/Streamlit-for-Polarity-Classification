# Streamlit for Toxicity Classification
This is a demo illustrating a classification model, and I trained the simple BERT model ("bert-base-uncased") on a [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) dataset. I did not train the BERT model with the whole dataset, instead, I used 1 persent only as this is not my main goal anyway. To put it succinctly, this BERT model can detect toxicity across a diverse range of conversations.

## Step 1: Download repository
Download my repository.

1. `git clone https://github.com/penguinwang96825/Streamlit-for-Toxicity-Classification.git toxicityClassification`
2. `cd toxicityClassification`

## Step 2: Download trained model from Google Cloud
- The toxicity classification model is trained using BERT.
1. [toxicity classifier model](https://drive.google.com/open?id=1plGGEs7__FnEfKTcdJm_1eitv5QiSUEg)
2. [toxicity classifier preprocessor](https://drive.google.com/open?id=12M-1dbC_C4iZhyvZ1X_NlGDG0GqOSnhe)

- Download the two files above and put them into the `toxicityClassification` folder.

## Step 3: Set up the environment
Download the requirements.txt file and run it in the terminal.

`pip install -r requirements.txt`

## Step 4: Run the app
1. Run the demo in terminal: `streamlit run app.py`
2. View the Streamlit app in your browser: `http://localhost:8501`

## © Copyright
See [License](https://github.com/penguinwang96825/Streamlit-for-Toxicity-Classification/blob/master/LICENSE) for more details.
