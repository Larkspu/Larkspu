from flask import Flask, render_template, request, redirect, url_for, session
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import torch
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key_here'  # Set the secret key

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    grade = db.Column(db.String(50), nullable=False)
    school = db.Column(db.String(100), nullable=False)

    # Define relationship with AnswerData
    answer_data = db.relationship('AnswerData', backref='user', lazy=True)

class AnswerData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Define user_id column explicitly
    question = db.Column(db.String(255), nullable=False)
    context = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)


tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Remaining code...



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'txt'

def get_top_answers(possible_starts, possible_ends, input_ids):
    answers = []
    for start, end in zip(possible_starts, possible_ends):
        # +1 for en
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end + 1]))
        answers.append(answer)
    return answers  

def answer_question(question, context, topN):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    
    input_ids = inputs["input_ids"].tolist()[0]

    model_out = model(**inputs)
     
    answer_start_scores = model_out["start_logits"]
    answer_end_scores = model_out["end_logits"]

    possible_starts = np.argsort(answer_start_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
    possible_ends = np.argsort(answer_end_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
    
    
    answer_start = torch.argmax(answer_start_scores)  
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    answers = get_top_answers(possible_starts, possible_ends, input_ids )

    return {
        "answer": answer,
        "answer_start": answer_start,
        "answer_end": answer_end,
        "input_ids": input_ids,
        "answer_start_scores": answer_start_scores,
        "answer_end_scores": answer_end_scores,
        "inputs": inputs,
        "answers": answers,
        "possible_starts": possible_starts,
        "possible_ends": possible_ends
    }

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        grade = request.form['grade']
        school = request.form['school']

        # Check if username already exists
        if User.query.filter_by(username=username).first():
            return "Username already exists. Please choose another username."

        # Create new user
        new_user = User(username=username, password=password, name=name, grade=grade, school=school)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()

        if user:
            # Set the username in the session upon successful login
            session['username'] = username
            
            # Redirect to dashboard or home page
            return redirect(url_for('query'))
        else:
            return "Invalid username or password. Please try again."

    return render_template('login.html')


@app.route('/query')
def query():
    return render_template('query.html')


@app.route('/answer', methods=['POST'])
def answer():
    if 'username' not in session:
        return redirect(url_for('login'))  

    question = request.form.get('question')
    context_file = request.files.get('context_file')
    username = session['username']  # Retrieve username from the session

    if context_file and allowed_file(context_file.filename):
        context = context_file.read().decode('utf-8')
    else:
        context = """
        Japan is the eleventh-most populous country in the world, as well as one of the most densely populated and urbanized.
        """

    # Get the user object from the database based on the username
    user = User.query.filter_by(username=username).first()

    if user:
        # Process the question and context to get the answer
        topN = 5  
        answer_map = answer_question(question, context, topN)

        # Store the answer along with the question and context
        answer = answer_map['answer']

        # Create a new AnswerData object and associate it with the user and the question
        
        answer_data = AnswerData( question=question, context=context, answer=answer)
        db.session.add(answer_data)
        db.session.commit()

        # Optional: You can retrieve the answer data associated with the user
        user_answer_data = user.answer_data

    return render_template('result.html', question=question, context=context, answer_map=answer_map)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)