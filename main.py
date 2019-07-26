from flask import Flask, render_template, request, redirect, url_for, flash, g
from flask_mongoengine import MongoEngine, Document
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.fields.html5 import EmailField
from wtforms.validators import Email, Length, InputRequired, EqualTo, DataRequired
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pickle
# ==============================================================================
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat

import torch
import torch.nn.functional as F

from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments
from utils import get_dataset_personalities, download_pretrained_model

import  interact
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==============================================================================

app = Flask(__name__)

app.config['MONGODB_SETTINGS'] = {
	'db': 'skylah-flask',
	'username': 'eshban',
	'password': 'Mettagross1',
	'host': 'mongodb://eshban:Mettagross1@ds221115.mlab.com:21115/skylah-flask'
}

db = MongoEngine(app)
app.config['SECRET_KEY'] = 'our-secret-key'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
print('Succesfully connected to MongoDB')

class User(UserMixin, db.Document):
	meta = {'collection': 'User'}
	name = db.StringField(max_length=50)
	email = db.StringField(max_length=30)
	password = db.StringField()
	#confirm = db.StringField()

'''
class Session(db.Document):
	meta:{'collection': 'Session'}
	startTime = db.DateTimeField()
	endTime = db.DateTimeField()

class Chat(db.Document):
	meta = {'collection':'Chat'}
	message = db.StringField(max_length=100)
	message_time = db.DateTimeField()
	response = db.StringField(max_length=100)
	response_time = db.DateTimeField()
'''
# ==============================================================================

@login_manager.user_loader
def load_user(user_id):
	return User.objects(pk=user_id).first()

class RegisterationForm(FlaskForm):
	name = StringField('Name', validators=[InputRequired(), Length(max=50)])

	email = EmailField('Email', validators=[DataRequired(), 
	Email()])
	
	password = PasswordField('Password', validators=[InputRequired(), EqualTo('confirm', message='Passwords must match'), Length(min=8, max=20)])

	confirm = PasswordField('Confirm Password')

	submit = SubmitField("Sign Up")

class LoginForm(FlaskForm):
	email = EmailField('Email', validators=[DataRequired(), 
	Email()])

	password = PasswordField('Password', validators=[InputRequired(), 
	Length(min=8, max=20)])

	submit = SubmitField("Login")

class ChatForm(FlaskForm):
	chatReply = ''
	newUser = True
	#chatInput = StringField('Name', validators=[InputRequired()])
	chatInput = TextAreaField('ChatInput')
	submit = SubmitField("Send")

# =====================================================================================

# ======================================================================================

def loader():
    f = open("checkpoints/personality.pickle", 'rb')
    personality = pickle.load(f)
    f.close()

    f = open("checkpoints/model.pickle", 'rb')
    model = pickle.load(f)
    f.close()

    f = open("checkpoints/tokenizer.pickle", 'rb')
    tokenizer = pickle.load(f)
    f.close()

    f = open("checkpoints/args.pickle", 'rb')
    args = pickle.load(f)
    f.close()

    return personality, model, tokenizer, args
# ===================================================================================

def chat_run(raw_text):

    personality, model, tokenizer, args = loader()

    history = []
    inputs = []
    outputs = []

    inputs.append(raw_text)

    if raw_text.lower() == 'bye':
        print("Take care. Bye!")

    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = interact.sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2*args.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    outputs.append(out_text)

    return out_text
    
   

# ======================================================================================

@app.route('/')
def landing():
	return render_template('landing.html')

@app.route('/objective')
def objective():
	return render_template('objective.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/join')
def join():
	return render_template('join.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
	form = RegisterationForm()
	#print('Formed Initialized')
	#print(form.data)
	#print("Register: ", request.query_string)
	#print(request.method)
	if request.method == 'POST':
		if form.validate():
			existing_user = User.objects(email=form.email.data).first()
			if existing_user is None:
				hashpass = generate_password_hash(form.password.data, method='sha256')
				print(form.email.data)
				user = User(form.name.data, form.email.data, hashpass).save()
				login_user(user)
				return redirect(url_for('dashboard'))
	return render_template('/register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
	if current_user.is_authenticated == True:
		return redirect(url_for('dashboard'))
	form = LoginForm(request.form)
	#print(request.method)
	if request.method == 'POST':
		if form.validate():
			check_user = User.objects(email = form.email.data).first()
			if check_user is not None:
				if check_password_hash(check_user['password'], form.password.data):
					login_user(check_user)
					g.user = check_user
					print(g.user.name)
					return redirect(url_for('dashboard'))
	return render_template('login.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
	return render_template('dashboard.html', name=current_user.name)

@app.route('/logout', methods=['GET'])
@login_required
def logout():
	logout_user()
	return redirect(url_for('login'))

global userInput, bot_response

userInput = list()
bot_response = list()

u = ['Hi', 'How are you?', 'What are you doing?', 
'Ah, okay!', 'I am unemployed actually', 
"I always stay at home. I know it is bad for health but who even cares about me!", 
"I had a dog, it's dead now. I loved it so much!", 
"Nah. Not really. My job takes a lot of time. It's so boring!"]

b = ['Hello', 'I am fine and you?', 'Nothing exciting actually!', 
'What do you do for a living?', 'That is sad to hear', 
"I always stay at home. I know it is bad for health but who even cares about me!", 
"I had a dog, it's dead now. I loved it so much!", 
"Nah. Not really. My job takes a lot of time. It's so boring!"]

for _ in u:
	userInput.append(_)

for _ in b:
	bot_response.append(_)

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
	chat_form = ChatForm()
	if request.method == 'POST':
		user_input = chat_form.chatInput.data
		print(user_input)
		userInput.append(user_input)
		#print(len(userInput))
		reply = chat_run(user_input)
		print(reply)
		bot_response.append(reply)
		#print(len(bot_response))
	chat_form.chatInput.data = ''
	return render_template('chat-1.html', form=chat_form, inputs=userInput, responses=bot_response)

def lookup(value):
	data = {0:"Happy", 1:"Happy", 2:"Happy", 3:"Quite Happy", 4:"Not Depressed", 5:"Not Depressed", 
	6:"Mildly Depressed", 7:"Mildly Depressed", 8:"Depressed", 9:"Highly Depressed"}

	value = round(value,1)*10
	print("VALUE->", value)
	if value in data:
		label = data[value]

	return label

def depression_detect(results):
	act_result = []
	labels = []
	cumm_score = 0

	for result in results:
		for _ in result:
			cumm_score = cumm_score + _ 
			act_result.append(round(_, 2))
			labels.append(lookup(_))
	cumm_score = cumm_score/len(act_result)
	return act_result, labels, cumm_score

@app.route('/report', methods=['GET'])
@login_required
def report():
    f = open("checkpoints/classifier.pickle", 'rb')
    clf = pickle.load(f)
    f.close()

    f = open("checkpoints/clf_tokenizer.pickle", 'rb')
    tokenizer_obj = pickle.load(f)
    f.close()

    test_samples_tokens = tokenizer_obj.texts_to_sequences(userInput)
    test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=100)

    results = clf.predict(x=test_samples_tokens_pad)
    results = results.tolist()
    results_, labels, score = depression_detect(results)
    verdict = lookup(score)
    return render_template('report.html', results = results_, userInput=userInput, labels=labels, score=score, verdict=verdict)

@login_manager.unauthorized_handler
def unauthorized_handler():
    return 'Unauthorized'

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
	app.run(debug=True, port=5002)