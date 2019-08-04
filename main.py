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

from response_generation import generate_unique_response
# ==============================================================================

#from app.chatbot.chatbot import sample_sequence, top_filtering, loader, chat_run

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
	#session = Session()
	start_time = str(datetime.now())

	chat_form = ChatForm()

	if request.method == 'POST':
		m_t = str(datetime.now())
		messages_time.append(m_t)

		user_input = chat_form.chatInput.data
		messages.append(user_input)

		if user_input.lower() == 'bye':
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
		session = Session()
		chat = Chat()
		chat.message = messages
		chat.message_time = messages_time
		chat.response = responses
		chat.response_time = responses_time
		chat.save()
		session.start_time = start_time
		session.end_time = end_time
		session.chat = chat
		current_user.session.append(session)
		current_user.save()
		return redirect(url_for('dashboard'))
	chat_form.chatInput.data = ''
	return render_template('chat.html', form=chat_form, inputs=messages, responses=responses)


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
    f = open("models/classifier/classifier.pickle", 'rb')
    clf = pickle.load(f)
    f.close()

    f = open("models/classifier/clf_tokenizer.pickle", 'rb')
    tokenizer_obj = pickle.load(f)
    f.close()
    '''
    test_samples_tokens = tokenizer_obj.texts_to_sequences(userInput)
    test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=100)

    results = clf.predict(x=test_samples_tokens_pad)
    results = results.tolist()
    results_, labels, score = depression_detect(results)
    verdict = lookup(score)
    return render_template('report.html', results = results_, userInput=userInput, labels=labels, score=score, verdict=verdict)
	'''
    return render_template('report.html')

@login_manager.unauthorized_handler
def unauthorized_handler():
    return 'Unauthorized'

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
	app.run(debug=True, port=5002)