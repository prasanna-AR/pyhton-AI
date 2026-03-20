from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model
class Todo(db.Model):
    task_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    done = db.Column(db.Boolean, default=False)

# Home Page
@app.route('/')
def home():
    # Latest first (history style)
    todo_list = Todo.query.order_by(Todo.task_id.desc()).all()
    return render_template('index.html', todo_list=todo_list)

# Add Task
@app.route('/add', methods=['POST'])
def add():
    name = request.form.get("name")

    if name:  # avoid empty input
        new_task = Todo(name=name)
        db.session.add(new_task)
        db.session.commit()

    return redirect(url_for("home"))

# Run App
if __name__ == '__main__':
    with app.app_context():
        db.create_all()   # create DB if not exists
    app.run(debug=True)