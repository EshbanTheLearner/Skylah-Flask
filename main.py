from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mongoengine import MongoEngine, Document
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.fields.html5 import EmailField
from wtforms.validators import Email, Length, InputRequired, EqualTo, DataRequired
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from markupsafe import Markup
import asyncio
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
	#chatInput = StringField('Name', validators=[InputRequired()])
    chatInput = TextAreaField('ChatInput')
    #submit_value = Markup('<span class="oi oi-check" title="Submit"><i class="material-icons right">send</i></span>')
    submit = SubmitField("Send")
    chatReply = ''
    

# ======================================================================================

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

async def prepare_chatbot():
    print("Starting to prepare chatbot")
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="finetuned_chatbot_gpt/", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_pretrained_model()
        #print("Did Not Worked, Bitch!")

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    logger.info("Sample a personality")
    personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    print("Type of personalities-> ", type(personalities))
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
    print("Preprations Complete")
    #await asyncio.sleep()
    return personality, tokenizer, model, args

def chat_run(raw_text, personality, tokenizer, model, args):

    history = []

    if raw_text.lower() == 'bye':
        print("Take care. Bye!")

    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2*args.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
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

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    chat_form = ChatForm()
    #personality, tokenizer, model, args = asyncio.run(prepare_chatbot())
    #render_template("wait.html")
    if request.method == 'POST':
        user_input = chat_form.chatInput.data
        print(user_input)
        #reply = chat_run(user_input, personality, tokenizer, model, args)
        reply = 'Hello from the other side'
        chat_form.chatReply = reply
        #print(reply)
    return render_template('chat.html', form=chat_form)


if __name__ == '__main__':
	app.run(debug=True, port=5002)