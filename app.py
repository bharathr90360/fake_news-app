# -*- coding: utf-8 -*-
from flask import Flask,render_template,request
import pickle
import re
import spacy
import string
nlp=spacy.load('en_core_web_sm')
import nltk




classifier=pickle.load(open('svm.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

app=Flask(__name__)

def preprocess(tweet):
  

    #HappyEmoticons
    emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
  
    # Sad Emoticons
    emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
      ])
  
    #combine sad and happy emoticons
    emoticons = emoticons_happy.union(emoticons_sad)


    #Emoji patterns
    emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)
    
    print(stop_words)
    word_tokens = tweet.split()
    #after tweepy preprocessing the colon symbol left remain after      #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    

    doc=nlp(tweet)
    filtered_words=[]
    for token in doc:  
      if ((token.is_punct==False) and (token.is_stop==False) and (token not in emoticons) and (token.is_space==False) and (token.is_ascii==True)):
        filtered_words.append(token.text.lower())


    return ' '.join(filtered_words)




@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = news
        
        cleaned_data=preprocess(data)
        vect = tfidf.transform([cleaned_data])
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    	
	app.run(debug=True)
