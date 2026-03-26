from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from dotenv import load_dotenv
import markdown
from datetime import datetime

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------------- LOAD ENV ---------------- #
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None

# ---------------- FLASK ---------------- #
app = Flask(__name__)
app.secret_key = "super_secret_study_buddy_key"
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs("uploads", exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- DATABASE ---------------- #
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    conversations = db.relationship('Conversation', backref='owner', lazy=True)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="New Chat")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='conversation', cascade="all, delete-orphan", lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    user_input = db.Column(db.Text)
    bot_response = db.Column(db.Text)

# ---------------- AUTH DECORATOR ---------------- #
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- PDF PROCESS ---------------- #
def process_pdf(filepath):
    global vectorstore
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(docs, embeddings)

# ---------------- PROMPT ---------------- #
def build_prompt(user_input, context, mode):
    if mode == "eli5":
        mode_text = "Explain this as if I'm 5 years old, using very simple words and analogies."
    elif mode == "example":
        mode_text = "Explain this clearly by providing a practical, real-life example."
    elif mode == "story":
        mode_text = "Explain this topic by telling an engaging, relatable real-life story."
    elif mode == "quiz":
        mode_text = "Generate a short, fun quiz (3 short questions) with answers at the end."
    elif mode == "notes":
        mode_text = "Convert this topic into concise Exam Notes. Include: Key points, Definitions, and Short answers."
    elif mode == "5min":
        mode_text = "Provide a high-yield, 5-minute revision summary in strictly 5 lines."
    elif mode == "random":
        mode_text = "Give me a random fun 'Brain Challenge' or 'Question of the day' related to this topic."
    else:
        mode_text = "Explain this clearly."

    return f"""
You are a helpful and strict teacher.

RULES:
- Step-by-step explanation
- Use Markdown formatting (**bold**, bullet points, and headings) to make it very easy to read
- Short sentences and clear structure
- No emojis

{mode_text}

Context:
{context}

Question:
{user_input}
"""

# ---------------- ROUTES ---------------- #

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password", "error")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first():
            flash("User already exists", "error")
        else:
            hashed = generate_password_hash(password)
            new_user = User(username=username, email=email, password_hash=hashed)
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@app.route('/c/<int:convo_id>')
@login_required
def home(convo_id=None):
    user_id = session['user_id']
    username = session['username']
    
    # Load user's conversations for sidebar
    conversations = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.updated_at.desc()).all()
    
    active_convo = None
    chats = []
    
    if convo_id:
        active_convo = Conversation.query.filter_by(id=convo_id, user_id=user_id).first()
        if active_convo:
            chats = Message.query.filter_by(conversation_id=convo_id).all()
            session['current_convo_id'] = convo_id

    return render_template('base.html', 
                         username=username,
                         conversations=conversations, 
                         chats=chats, 
                         active_convo_id=convo_id)

@app.route('/new_chat')
@login_required
def new_chat():
    session.pop('current_convo_id', None)
    return redirect(url_for('home'))

@app.route('/delete_convo/<int:convo_id>', methods=['POST'])
@login_required
def delete_convo(convo_id):
    convo = Conversation.query.filter_by(id=convo_id, user_id=session['user_id']).first()
    if convo:
        db.session.delete(convo)
        db.session.commit()
        if session.get('current_convo_id') == convo_id:
            session.pop('current_convo_id', None)
    return jsonify({"status": "deleted"})


@app.route('/upload_pdf', methods=['POST'])
@login_required
def upload_pdf():
    file = request.files.get('pdf')
    if file:
        path = os.path.join("uploads", file.filename)
        file.save(path)
        process_pdf(path)
        return jsonify({"message": "PDF uploaded!"})
    return jsonify({"error": "Upload failed"}), 400


@app.route('/ask', methods=['POST'])
@login_required
def ask():
    global vectorstore
    user_input = request.form.get("prompt")
    mode = request.form.get("mode", "normal")
    user_id = session['user_id']
    # Get active conversation or create one
    convo_id = session.get('current_convo_id')
    if not convo_id:
        # Generate title from first few words
        title = " ".join(user_input.split()[:5]) + "..." if user_input else "New Chat"
        convo = Conversation(user_id=user_id, title=title)
        db.session.add(convo)
        db.session.commit()
        convo_id = convo.id
        session['current_convo_id'] = convo_id
    else:
        convo = Conversation.query.get(convo_id)
        convo.updated_at = datetime.utcnow()
        db.session.commit()

    if vectorstore is None:
        # simple fallback if no pdf uploaded
        reply = llm.invoke(user_input).content
    else:
        docs = vectorstore.similarity_search(user_input, k=5)
        context = "\n".join([d.page_content for d in docs])
        prompt = build_prompt(user_input, context, mode)
        response = llm.invoke(prompt)
        reply = response.content

    bot_html = markdown.markdown(reply, extensions=['fenced_code', 'tables'])

    msg = Message(conversation_id=convo_id, user_input=user_input, bot_response=bot_html)
    db.session.add(msg)
    db.session.commit()

    return jsonify({
        "msg_id": msg.id,
        "user": user_input, 
        "bot": bot_html, 
        "convo_id": convo_id,
        "new_convo": len(msg.conversation.messages) == 1,
        "title": convo.title
    })

@app.route('/edit_msg/<int:msg_id>', methods=['POST'])
@login_required
def edit_msg(msg_id):
    global vectorstore
    new_input = request.form.get("prompt")
    mode = request.form.get("mode", "normal")
    msg = Message.query.get_or_404(msg_id)
    
    if msg.conversation.user_id != session['user_id']:
        return jsonify({"error": "Unauthorized"}), 403

    if vectorstore is None:
        reply = llm.invoke(new_input).content
    else:
        docs = vectorstore.similarity_search(new_input, k=5)
        context = "\n".join([d.page_content for d in docs])
        prompt = build_prompt(new_input, context, mode)
        response = llm.invoke(prompt)
        reply = response.content

    bot_html = markdown.markdown(reply, extensions=['fenced_code', 'tables'])
    
    msg.user_input = new_input
    msg.bot_response = bot_html
    db.session.commit()

    return jsonify({"bot": bot_html})

@app.route('/clear', methods=['POST'])
@login_required
def clear():
    convo_id = session.get('current_convo_id')
    if convo_id:
        Message.query.filter_by(conversation_id=convo_id).delete()
        db.session.commit()
    return jsonify({"status": "cleared"})

# ---------------- RUN ---------------- #
if __name__ == '__main__':
    with app.app_context():
        # WARNING: Dropping tables deletes old chats!
        # db.drop_all()
        db.create_all()
    app.run(debug=True)