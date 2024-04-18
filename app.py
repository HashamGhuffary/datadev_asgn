from typing import Optional, List

from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import os

from keras.src.saving import load_model
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask_sqlalchemy import SQLAlchemy
import cv2
from flask_migrate import Migrate

app = Flask(__name__)

model = load_model('identifier_model.h5')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database file
db = SQLAlchemy(app)
migrate = Migrate(app, db)

label_dict = {
    "1": "Mild",
    "2": "Moderate",
    "3": "No_DR",
    "4": "Proliferate_DR",
    "5": "Severe",
}

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    account_type = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.full_name


# User.metadata.create_all(app)

# Function to preprocess image before prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Apply any necessary preprocessing steps
    return image


# Define a function to classify the image
def classify_image(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    result = model.predict(img)
    return result


# Route for home page
@app.route('/')
def home():
    return render_template('landing_page/index.html')


@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        email_or_username = request.form['email_or_username']
        password = request.form['password']

        # Query the database for a user with the provided email or username
        user = User.query.filter((User.email == email_or_username) | (User.full_name == email_or_username)).first()

        if user and user.password == password:
            # Validation succeeded, redirect to a success page
            # For now, let's redirect to the home page
            return redirect(url_for('home'))
        else:
            # Validation failed, render login page with an error message
            error = "Invalid email/username or password."
            return render_template('login page/index.html', error=error)

    return render_template('login page/index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        account_type = request.form['account_type']

        # Create a new User object
        new_user = User(full_name=full_name, email=email, password=password, account_type=account_type)

        # Add the new user to the database
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('signup_success'))
    return render_template('sign-uppage1/index.html')


# Route to handle image upload and prediction
@app.route('/upload', methods=['POST', "GET"])
def upload():
    if 'image' not in request.files:
        return render_template('upload.html', result="No image file uploaded.")
    file = request.files['image']
    if file.filename == '':
        return render_template('upload.html', result="No selected image file.")
    if file:
        image_path = os.path.join('uploads', file.filename)
        # file.save(image_path)
        # result = classify_image(image_path)
        # # Assuming 'result' is a class label or probability score
        # # You can format the result as desired
        # return render_template('upload.html', result=result)

        img = Image.open(file)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        predicted_label = np.argmax(prediction)
        return render_template('upload.html', result=label_dict[str(predicted_label)])


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)
