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
from whitenoise import WhiteNoise
import nltk
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

from flask import Response
import matplotlib; matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from response_generation import generate_unique_response
from utils import lookup, depression_detect, load_clf
from report_gen import detect, freq_words, clean_text, depression_dist, depression_trend
# ==============================================================================

#from app.chatbot.chatbot import sample_sequence, top_filtering, loader, chat_run
from app.chatbot.chatbot import chat_run

# ==============================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================

import config

# ==============================================================================
app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
app.config['MONGODB_SETTINGS'] = config.db_credentials

# IMPORTANT TODOS
'''
Managing database objects seperatly
'''

db = MongoEngine(app)
#db = MongoEngine()
app.config['SECRET_KEY'] = 'our-secret-key'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = "strong"
print('Succesfully connected to MongoDB')
'''
class Report(db.Document):
	meta = {'collection': 'Report'}
	word_frequency_plot = db.FileField()
	depression_distribution = db.FileField()
	depression_trend = db.FileField()
'''
class Chat(db.Document):
	meta = {'collection': 'Chat'}
	message = db.ListField(db.StringField())
	response = db.ListField(db.StringField())
	message_time = db.ListField(db.StringField())
	response_time = db.ListField(db.StringField())

class Session(db.EmbeddedDocument):
	meta = {'collection': 'Session'}
	start_time = db.StringField()
	end_time = db.StringField()
	chat = db.ReferenceField(Chat)
	#report = db.ReferenceField(Report)
	total_score = db.FloatField()
	score_per_sentence = db.ListField(db.FloatField()) # array of score per sentence

class User(UserMixin, db.Document):
	meta = {'collection': 'User'}
	name = db.StringField(max_length=50)
	email = db.StringField(max_length=30)
	password = db.StringField()
	session = db.EmbeddedDocumentListField(Session)

# ==============================================================================

@login_manager.user_loader
def load_user(user_id):
	return User.objects(pk=user_id).first()

class RegisterationForm(FlaskForm):
	name = StringField('Name', validators=[InputRequired(), Length(max=50)])

	email = EmailField('Email', validators=[DataRequired(), Email()])

	errors_email = []
	
	password = PasswordField('Password', validators=[InputRequired(), 
	EqualTo('confirm', message='Passwords must match'), Length(min=8, max=20)])

	confirm = PasswordField('Confirm Password')

	errors_confirm_password = []

	submit = SubmitField("Sign Up")

class LoginForm(FlaskForm):
	email = EmailField('Email', validators=[DataRequired(), 
	Email()])

	errors_email = []

	password = PasswordField('Password', validators=[InputRequired(), 
	Length(min=8, max=20)])

	errors_password = []

	submit = SubmitField("Login")

class ChatForm(FlaskForm):
	chatReply = ''
	newUser = True
	#chatInput = StringField('Name', validators=[InputRequired()])
	chatInput = TextAreaField('ChatInput')
	submit = SubmitField("Send")

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
	form.errors_email = []
	form.errors_confirm_password = []
	if request.method == 'POST':
		if form.validate():
			existing_user = User.objects(email=form.email.data).first()
			if existing_user is None:
				hashed_password = generate_password_hash(form.password.data, method='sha256')

				user = User(form.name.data, form.email.data, hashed_password).save()
				
				login_user(user)
				return redirect(url_for('dashboard'))
			else:
				form.errors_email.append("Email already exists")
		else:
			form.errors_confirm_password.append("Passwords must match")
	return render_template('/register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
	if current_user.is_authenticated == True:
		return redirect(url_for('dashboard'))
	form = LoginForm(request.form)
	form.errors_email = []
	form.errors_password = []
	if request.method == 'POST':
		if form.validate():
			check_user = User.objects(email = form.email.data).first()
			if check_user is not None:
				if check_password_hash(check_user['password'], form.password.data):
					login_user(check_user)
					g.user = check_user
					print(g.user.name)
					return redirect(url_for('dashboard'))
				else:
					form.errors_password.append("Inccorect password")
			else:
				form.errors_email.append("Email not registered")
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

messages = []
messages_time = []
responses = []
responses_time = []
start_time = ''
end_time = ''

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
	break_op = False
	triggers = ["bye", "goodbye", "i gotta go now", "gotta go now", "take care"]
	#session = Session()
	#start_time = str(datetime.now())
	chat_form = ChatForm()
	if request.method == 'POST':
		m_t = str(datetime.now())
		messages_time.append(m_t)

		user_input = chat_form.chatInput.data
		user_input = user_input.strip()
		print(type(user_input))
		print(user_input)
		messages.append(user_input)

		if user_input.lower() in triggers:
			print("THIS IS THE END OF WAKANDA")
			end_time = str(datetime.now())
			break_op = True

		print(m_t, " : ", user_input)
		reply = generate_unique_response(user_input)
		
		#reply = chat_run(user_input)
		r_t = str(datetime.now())
		responses_time.append(r_t)

		print(r_t, " : ", reply)
		responses.append(reply)
	
	if break_op:
		end_time = str(datetime.now())
		print("Saving session in DB...")
		session = Session()
		chat = Chat()
		chat.message = messages
		chat.message_time = messages_time
		chat.response = responses
		chat.response_time = responses_time
		chat.save()
		session.start_time = start_time
		session.end_time = end_time
		session.score_per_sentence =  None
		session.total_score = None
		session.chat = chat
		current_user.session.append(session)
		current_user.save()
		generate_and_save_report()
		return redirect(url_for('dashboard'))
	
	chat_form.chatInput.data = ''
	return render_template('chat.html', form=chat_form, inputs=messages, responses=responses)

def generate_and_save_report():
	messages = []
	messages = current_user.session[-1].chat.message
	print(messages)
	input_tensor = [[m] for m in messages]
	print(input_tensor)
	print("Calculating scores...")
	results_, score_latest = detect(input_tensor)
	print(score_latest)
	print(results_)	
	current_user.session[-1].total_score = score_latest
	current_user.session[-1].score_per_sentence = results_
	current_user.save()
	print("No of sessions:", len(current_user.session))
	
def total_report():
	total_score = 0.0
	total_scores_array = []
	total_messages_array = []
	for i in range(0, len(current_user.session)):
		if current_user.session[i].total_score is None:
			total_score += 0.0
		total_score += current_user.session[i].total_score
		total_scores_array += current_user.session[i].score_per_sentence
		total_messages_array += current_user.session[i].chat.message
	total_score /= len(current_user.session)
	x = np.arange(len(total_scores_array))
	total_text = " ".join(m for m in total_messages_array)
	total_text = clean_text(total_text)

	return  total_score, x, total_scores_array, total_text

def latest_report():
	score = 0.0
	scores_array = []
	messages = []
	
	score = current_user.session[-1].total_score
	#score += score
	scores_array += current_user.session[-1].score_per_sentence
	messages += current_user.session[-1].chat.message
	score = score / len(current_user.session)
	x = np.arange(len(scores_array))
	text = " ".join(m for m in messages)
	text = clean_text(text)

	return  score, x, scores_array, text

@app.route('/report_latest', methods=['GET'])
@login_required
def report_latest():
	if len(current_user.session) <= 0:
		flash('You need to chat in order to generate a report')
		return redirect(url_for('dashboard'))
	score, x, scores_array, text = latest_report()
	url1 = freq_words(text)
	url2 = depression_dist(scores_array)
	url3 = depression_trend(x, scores_array)
	verdict = lookup(score)
	return render_template('report_latest.html', score=round(score, 2), verdict=verdict, plot1=url1, plot2=url2, plot3=url3)


@app.route('/report', methods=['GET'])
@login_required
def report():
	if len(current_user.session) <= 0:
		flash('You need to chat in order to generate a report')
		return redirect(url_for('dashboard'))
	total_score, x, total_scores_array, total_text = total_report()
	url1 = freq_words(total_text)
	url2 = depression_dist(total_scores_array)
	url3 = depression_trend(x, total_scores_array)

	verdict = lookup(total_score)

	return render_template('report.html',score=round(total_score, 2), verdict=verdict, plot1=url1, plot2=url2, plot3=url3)
	
@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect('/')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
	app.run(debug=True, port=5002)