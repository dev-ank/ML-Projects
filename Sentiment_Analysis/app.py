from flask import Flask, render_template, flash, request, url_for, request
import numpy as np
import pandas as pd
import re
import os
from tensorflow.python.keras.backend import set_session
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from tensorflow.python.keras.models import load_model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

IMAGE_FOLDER=os.path.join('static','img_pool')

app=Flask(__name__)

app.config['UPLOAD_FOLDER']=IMAGE_FOLDER


sess = tf.compat.v1.Session()
set_session(sess)
model=load_model('Sentiment_saved.h5')
graph=tf.compat.v1.get_default_graph()


@app.route('/',methods=['GET','POST'])
def home():
	return render_template("home.html")

@app.route('/sentiment_analysis_prediction',methods=['GET','POST'])
def sent_any_pred():
	if request.method=='POST':
		text=request.form['text']
		Sentiment=""
		max_review_length=500
		word_to_index=imdb.get_word_index()
		strip_special_chars=re.compile("[^A-Za-z0-9 ]+")
		text=text.lower().replace("<br />"," ")
		text=re.sub(strip_special_chars,"",text.lower())

		words=text.split()
		x_test=[[word_to_index[word] if(word in word_to_index and word_to_index[word]<=5000) else 0 for word in words]]
		x_test=sequence.pad_sequences(x_test,maxlen=500)
		vector=np.array([x_test.flatten()])
		#print(x_test)

		with graph.as_default():
			set_session(sess)
			#print(sess)
			probability=model.predict(vector)
			class1=model.predict_classes(vector)
		if class1==0:
			Sentiment="Negative"
			img_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
		else:
			Sentiment="Positive"
			img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')

	return render_template('home.html', text=text, sentiment=Sentiment, probability=probability, image=img_filename)


if __name__=="__main__":
    app.run()