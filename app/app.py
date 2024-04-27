from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
import torch
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer
import os
from nltk.stem import PorterStemmer
from nltk.util import ngrams
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

MODEL_PATH = "../models/distil_bert_6_.pth"

class Classifier_V2(nn.Module):
    '''
    This model is similar to the first one but instead of using the pooler output, it uses the hidden states of the model
    The 'hidden_states_used' parameter is used to determine how many hidden states to use, smaller values of this will be less computationally expensive, but likely less accurate
    '''
    def __init__(self, model_name ,num_labels,hidden_states_used):
        super(Classifier_V2,self).__init__()
        self.hidden_states_used = hidden_states_used
        self.model = BertModel.from_pretrained(model_name,config = BertConfig.from_pretrained(model_name,output_hidden_states = True,num_labels=num_labels))
        self.hidden1 = nn.Linear(self.model.config.hidden_size*self.model.config.max_position_embeddings*self.hidden_states_used, 64)
        self.hidden_p = nn.Linear(self.model.config.hidden_size, 64)
        self.fc = nn.Linear(64, num_labels)
        self.dropout = nn.Dropout(0.1)
        if num_labels == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
    
    def forward(self, attention_mask = None, token_type_ids = None,input_ids = None):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        hidden_states = torch.cat(outputs.hidden_states[-self.hidden_states_used:],dim=0).view(1,-1)
        pooler_output = outputs.pooler_output
        x_pooler = self.hidden_p(self.dropout(pooler_output))
        x_hidden = self.hidden1(self.dropout(hidden_states))
        x = torch.add(x_pooler,x_hidden)
        output = self.fc(x)
        logit = self.activation(output)
        return logit
    

# model = Classifier_V2('distilbert/distilbert-base-uncased',1,6).to(device)
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()
# tokenizer = BertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
stemmer = PorterStemmer()
bow_model = joblib.load('../models/logistic_regression_bow.pkl')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
vectorizer = joblib.load('../models/bow_count_vectorizer.pkl')

def model_predict(text, model):
    # Prediction function for BERT model
    raw_preds = []
    for sentence in text:
        input_ids = torch.tensor(sentence['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(sentence['attention_mask']).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(sentence['token_type_ids']).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(attention_mask = attention_mask, token_type_ids = token_type_ids, input_ids = input_ids)
        raw_preds.append(preds.detach().cpu().numpy().tolist())
    return raw_preds

def tokenize_text(text,tokenizer):
    # Placeholder for text tokenization
    sentences = text.split('.')
    tokenized_sentences = [tokenizer(sentence,max_length=512,padding='max_length') for sentence in sentences]
    return tokenized_sentences


def preprocess(text, n=2):
    # text -> tokens
    tokens = word_tokenize(text)
    # stem and removing stopwords
    tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    # n-gram generation
    n_grams = list(ngrams(tokens, n))
    # flattening list of n-grams
    n_grams = ['_'.join(gram) for gram in n_grams]
    return tokens + n_grams

@app.route("/",methods = ['GET'])
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        #read the file and save it to the uploads folder 
        data = request.get_json(force=True)
        text = data['text']
        # print(text)
        tokenized_text = tokenize_text(text,tokenizer)
        #make a prediction
        # print(tokenized_text[0]['input_ids'])
        preds = model_predict(tokenized_text, model)
        #process the prediction to determine the output
        # print(preds)
        return jsonify(prediction=preds,text=text.split('.'))
    return None

@app.route("/predict_bow", methods = ['GET','POST'])
def predict_bow():
    if request.method == 'POST':
        preds = []
        data = request.get_json(force=True)
        texts = data['text']
        print(texts)
        preprocessed_text = [preprocess(text, n=2) for text in texts.split('.')]
        texts_joined = [' '.join(text) for text in preprocessed_text]
        print(texts_joined)
        vectorized_text = vectorizer.transform(texts_joined)
        preds = bow_model.predict(vectorized_text)
        print(preds)
        return jsonify(prediction=preds.tolist(),text=texts.split('.'))
    return None

if __name__ == '__main__':
    app.run(debug=True)