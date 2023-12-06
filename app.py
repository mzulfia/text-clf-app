import pickle
import streamlit as st
import pandas as pd
import sklearn
import nlp_id
import PIL
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_id.lemmatizer import Lemmatizer
import numpy as np
from PIL import Image
import re
import time

# Load data from the pickle file
model_file = 'text-classification-svm-tfidf.pickle'
with open(model_file, 'rb') as file:
    data = pickle.load(file)

tfidf_vectorizer = data['tfidf_vectorizer']
mdl_svm_main = data['mdl_svm_tfidf_main']
mdl_svm_sub = data['mdl_svm_tfidf_sub']
label_main = data['label_main']
label_sub = data['label_sub']

def casefolding(review):
  emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
  review = str(review).lower()
  review = review.strip()
  review = re.sub(r'[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', '', review) # remove punctuation
  review = emoji_pattern.sub(r'', review) # remove emoji
  review = re.sub(r'\d+', '', review) # remove number

  return review

def lemmatizing(review):
  lemmatizer = Lemmatizer()
  text = lemmatizer.lemmatize(review)
  return text

def tokenization(review):
  nstr = review.split(' ')
  dat = []
  a = -1
  for hu in nstr:
    a = a + 1
  if hu == '':
    dat.append(a)
  p = 0
  b = 0
  for q in dat:
    b = q - p
    del nstr[b]
    p = p + 1
  return nstr

def stopword_removal(review):
  dt_sw = pd.read_csv('stopwords.csv', low_memory=False, header=None)
  sw_ext = dt_sw[0].to_numpy()
#   list_stopwords = stopwords.words('indonesian')
#   list_stopwords.extend(sw_ext)

  x = []
  data = []
  def myFunc(x):
    if x in sw_ext:
      return False
    else:
      return True
  fit = filter(myFunc, review)
  for x in fit:
    if x != '':
      data.append(x)
  return data

def mergeWS(review):
  do = []
  for w in review:
    do.append(w)
  merged_wd =[]
  merged_wd =" ".join(do)
  return merged_wd


st.title('Sistem Automasi Pengklasifikasian Ulasan PLN Mobile')
st.markdown('Rifki Zamzammi (NIM. 23522301), Muhammad Zulfi Ashari (NIM. 23522304), Satria Dina Astari (NIM. 23522309)')

st.header("Mesin Inferensi")

with st.sidebar.header('Automation System'):
    logo_pln_itb = Image.open('./assets/logo-pln-itb.png')
    st.image(logo-pln-itb)
    st.sidebar.markdown('''Dikembangkan oleh **Kelompok 1 - IF5171**''')

with st.form('inference_form', clear_on_submit=True):
   review = st.text_input('Ulasan', placeholder='Masukkan ulasan yang ingin diinferensi')
   submitted = st.form_submit_button('Submit', type='primary')
        

if review != '':
  ISI_ULASAN = casefolding(review)
  ISI_ULASAN = lemmatizing(ISI_ULASAN)
  ISI_ULASAN = tokenization(ISI_ULASAN)
  ISI_ULASAN = mergeWS(ISI_ULASAN)
  input_text = [ISI_ULASAN]
  X_test = tfidf_vectorizer.transform(input_text)
  y_pred_main = mdl_svm_main.predict(X_test)
  y_pred_sub = mdl_svm_sub.predict(X_test)
    
  st.header('Hasil') 
  with st.spinner('Processing...'):
    time.sleep(1)    
    st.write(f'Isi Ulasan: **{review}**')
    st.write(f'Klasifikasi Fitur: **{label_main.inverse_transform(y_pred_main)[0]}**')
    st.write(f'Sub Klasifikasi Fitur: **{label_sub.inverse_transform(y_pred_sub)[0]}**')
    review = st.empty()
  
 
   

