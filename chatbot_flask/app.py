from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

#create the api model---------------------------------------
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key)

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model
class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(500))
    bot = db.Column(db.String(2000))

# Home Page
@app.route('/')
def home():
    chats = Chat.query.all()
    return render_template('base.html', chats=chats)

# Handle Prompt
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get("prompt")

    if user_input:
        try:
            response = llm.invoke(user_input)
            bot_reply = response.content
        except Exception as e:
            bot_reply = "Error: " + str(e)

        chat = Chat(user=user_input, bot=bot_reply)
        db.session.add(chat)
        db.session.commit()

    return redirect(url_for("home"))

# Run App
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)