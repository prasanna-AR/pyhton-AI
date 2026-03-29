from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import os
import base64
from dotenv import load_dotenv
import markdown
from datetime import datetime

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

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
os.makedirs(os.path.join("static", "uploads"), exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- DATABASE ---------------- #
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    avatar = db.Column(db.String(300), nullable=True)
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
def get_mode_text(mode):
    if mode == "eli5":
        return "Explain this as if I'm 5 years old, using very simple words and analogies."
    elif mode == "example":
        return "Explain this clearly by providing a practical, real-life example."
    elif mode == "story":
        return "Explain this topic by telling an engaging, relatable real-life story."
    elif mode == "quiz":
        return "Generate a short, fun quiz (3 short questions) with answers at the end. (Ignore the 300-word minimum for this)."
    elif mode == "notes":
        return "Convert this topic into concise Exam Notes. Include: Key points, Definitions, and Short answers."
    elif mode == "5min":
        return "Provide a high-yield, 5-minute revision summary in strictly 5 lines. (Ignore the 300-word minimum for this)."
    elif mode == "random":
        return "Give me a random fun 'Brain Challenge' or 'Question of the day' related to this topic."
    else:
        return "Explain this clearly."

def get_system_prompt(mode):
    mode_text = get_mode_text(mode)
    return f"""You are an engaging, helpful Study Buddy and strict teacher.

RULES:
- Length: Your response MUST be between 300 and 800 words, explaining the topic thoroughly.
- Structure: ALWAYS put each point on a NEW LINE. Add blank lines between paragraphs and list items. DO NOT pack multiple points into a single paragraph!
- Formatting: Use rich Markdown (**bold**, headings, code blocks) to make it highly attractive and easy to read.
- Lists: Use standard Markdown lists (- or *), but you MUST place a relevant emoji at the START of the text for EVERY list item (e.g., "- 🚀 Point 1", "- 💡 Point 2").
- Tone: Engaging, supportive, and easy to understand (like ChatGPT).

{mode_text}"""

def build_prompt(user_input, context, mode):
    rules = get_system_prompt(mode)
    return f"{rules}\n\nContext:\n{context}\n\nQuestion:\n{user_input}"

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
            session['avatar'] = user.avatar
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
            
            # Auto-login after register
            session['user_id'] = new_user.id
            session['username'] = new_user.username
            session['avatar'] = None
            flash("Registration successful.", "success")
            return redirect(url_for('home'))

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


@app.route('/upload_avatar', methods=['POST'])
@login_required
def upload_avatar():
    file = request.files.get('avatar')
    if file and file.filename != '':
        os.makedirs(os.path.join("static", "uploads", "avatars"), exist_ok=True)
        ext = file.filename.rsplit('.', 1)[-1].lower()
        filename = secure_filename(f"avatar_{session['user_id']}_{int(datetime.now().timestamp())}.{ext}")
        path = os.path.join("static", "uploads", "avatars", filename)
        file.save(path)
        
        user = User.query.get(session['user_id'])
        user.avatar = filename
        db.session.commit()
        session['avatar'] = filename
        return jsonify({"status": "success", "avatar_url": url_for('static', filename=f"uploads/avatars/{filename}")})
    return jsonify({"error": "No file uploaded"}), 400


@app.route('/ask', methods=['POST'])
@login_required
def ask():
    global vectorstore
    user_input = request.form.get("prompt", "")
    mode = request.form.get("mode", "normal")
    user_id = session['user_id']
    
    file = request.files.get("image")
    image_base64 = None
    saved_filename = None
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        path = os.path.join("static", "uploads", filename)
        file.save(path)
        saved_filename = filename
        with open(path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # Get active conversation or create one
    convo_id = session.get('current_convo_id')
    if not convo_id:
        title = " ".join(user_input.split()[:5]) + "..." if user_input.strip() else "Image Discussion"
        convo = Conversation(user_id=user_id, title=title)
        db.session.add(convo)
        db.session.commit()
        convo_id = convo.id
        session['current_convo_id'] = convo_id
    else:
        convo = Conversation.query.get(convo_id)
        convo.updated_at = datetime.utcnow()
        db.session.commit()

    # Fetch previous messages for context
    history = []
    if convo_id:
        past_msgs = Message.query.filter_by(conversation_id=convo_id).order_by(Message.id.asc()).all()
        for pm in past_msgs[-10:]:  # Keep last 10 messages
            history.append(HumanMessage(content=pm.user_input))
            history.append(AIMessage(content=pm.bot_response))

    rules = get_system_prompt(mode)
    prompt_text = f"{rules}\n\nQuestion:\n{user_input}"
    
    if vectorstore is not None and user_input.strip():
        docs = vectorstore.similarity_search(user_input, k=5)
        context = "\n".join([d.page_content for d in docs])
        prompt_text = f"{rules}\n\nContext:\n{context}\n\nQuestion:\n{user_input}"

    content = []
    content.append({"type": "text", "text": prompt_text})
    if image_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
        
    history.append(HumanMessage(content=content))
    reply = llm.invoke(history).content

    bot_html = markdown.markdown(reply, extensions=['fenced_code', 'tables'])

    db_user_input = user_input
    if saved_filename:
        db_user_input = f'<img src="/static/uploads/{saved_filename}" class="chat-image-preview"><br>{user_input}'

    msg = Message(conversation_id=convo_id, user_input=db_user_input, bot_response=bot_html)
    db.session.add(msg)
    db.session.commit()

    return jsonify({
        "msg_id": msg.id,
        "user": db_user_input, 
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

    # Fetch previous messages for context up to this message
    history = []
    past_msgs = Message.query.filter_by(conversation_id=msg.conversation_id).filter(Message.id < msg_id).order_by(Message.id.asc()).all()
    for pm in past_msgs[-10:]:  # Keep last 10 messages for context
        history.append(HumanMessage(content=pm.user_input))
        history.append(AIMessage(content=pm.bot_response))

    if vectorstore is None:
        rules = get_system_prompt(mode)
        prompt = f"{rules}\n\nQuestion:\n{new_input}"
        history.append(HumanMessage(content=prompt))
        reply = llm.invoke(history).content
    else:
        docs = vectorstore.similarity_search(new_input, k=5)
        context = "\n".join([d.page_content for d in docs])
        prompt = build_prompt(new_input, context, mode)
        history.append(HumanMessage(content=prompt))
        response = llm.invoke(history)
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
        try:
            db.session.execute(text("ALTER TABLE user ADD COLUMN avatar VARCHAR(300)"))
            db.session.commit()
        except Exception:
            db.session.rollback()
        db.create_all()
    app.run(debug=True)