from flask import Flask, render_template, request, redirect, url_for, flash
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from markupsafe import Markup
import numpy as np
import pandas as pd
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')
))
disease_model.eval()

def predict_image(img,model=disease_model):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])

        image= Image.open(io.BytesIO(img))
        img_t = transform(image)
        img_u = torch.unsqueeze(img_t,0)

        #now  we get prediciton from model
        yb= model(img_u)
        _,preds = torch.max(yb,dim=1)
        prediction = disease_classes[preds[0].item()]

        return  prediction

# Load the pickled model and preprocessor
with open('models/best_model.pkl', 'rb') as model_file, open('models/preprocessor.pkl', 'rb') as preprocessor_file:
    best_model = pickle.load(model_file)
    preprocessor = pickle.load(preprocessor_file)

#------------------------------------------------------#


app = Flask(__name__, static_url_path='/static')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/khetify'

# Additional configurations (optional)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable Flask-SQLAlchemy modification tracking
app.config['SQLALCHEMY_ECHO'] = True  # Print SQL queries to the console for debugging (optional)

db = SQLAlchemy(app)

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    __table_args__ = {'schema': 'khetify'}
    uid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    admin = db.relationship('Admin', back_populates='user', uselist=False)

    def get_id(self):
        return str(self.uid)



class Admin(UserMixin, db.Model):
    __tablename__ = 'admins'
    __table_args__ = {'schema': 'khetify'}
    adminid = db.Column(db.Integer, primary_key=True)
    userid = db.Column(db.Integer, db.ForeignKey('khetify.users.uid'))
    user = db.relationship('User', back_populates='admin')
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def get_id(self):
        return str(self.adminid)



class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')


login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Specify the login route

@login_manager.user_loader
def load_user(user_id):
    user_id = int(user_id)

    admin = Admin.query.filter_by(adminid=user_id).first()
    if admin:
        # Admin
        print(f"Loaded admin: {admin.username}")
        return admin

    user = User.query.get(user_id)
    if user:
        # Regular user
        print(f"Loaded user: {user.username}")
        return user

    print(f"User with ID {user_id} not found.")
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()

        if user:
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your username and password.', 'error')

    return render_template('login.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        print(f"Input: Username={username}, Password={password}")

        admin = Admin.query.filter_by(username=username, password=password).first()

        print(f"Database: Admin={admin}")

        if admin:
            print(f"Logged in as admin: {admin.username}")
            login_user(admin)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Admin login failed. Please check your username and password.', 'error')

    return render_template('admin-login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        # Form is valid, process registration
        username = form.username.data
        email = form.email.data
        password = form.password.data
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    # Form is not valid or not submitted yet, render registration template with form
    return render_template('registration.html', title='Register', form=form)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    print(f"Current user: {current_user}")
    print(f"Is authenticated: {current_user.is_authenticated}")
    print(f"Is instance of Admin: {isinstance(current_user, Admin)}")

    if current_user.is_authenticated and isinstance(current_user, Admin):
        # Query all users for display on the admin dashboard
        all_users = User.query.all()
        return render_template('admin.html', all_users=all_users)
    else:
        flash('You do not have permission to access the admin dashboard.', 'error')
        return redirect(url_for('login'))



@app.route('/admin/delete_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def delete_user(user_id):
    user_to_delete = User.query.get_or_404(user_id)

    # Perform user deletion
    db.session.delete(user_to_delete)
    db.session.commit()

    flash(f'User {user_to_delete.username} has been deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

# render Homepage
@app.route('/')
@login_required
def home():
    title = 'Khetify - Home'
    return render_template('index.html',title=title)

#render crop yield prediction from page
@ app.route('/crop')
@login_required
def crop_predict():
    title = 'Khetify - Crop Yield Prediction'
    return render_template('crop.html', title=title)

#render fertilizer recommendation form page
@app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = ' Khelify - Fertilizer Recommendation'

    return render_template('fertilizer.html',title=title)

#render about page
@app.route('/about')
@login_required
def about():
    return render_template('about.html')

#render aboutcrop
@app.route('/about_crop')
@login_required
def about_crop():
    return render_template('about-crop.html')

#render aboutfer
@app.route('/about_fer')
@login_required
def about_fer():
    return render_template('about-fer.html')

#render aboutplant
@app.route('/about_plant')
@login_required
def about_plant():
    return render_template('about-plant.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))
#--------------------------------------------------#

#RENDER PREDICTION PAGES

#render crop yield prediction page
@ app.route('/crop-prediction', methods=['POST'])
def crop_prediction():
    title = 'Khetify- Crop Recommendation'

    if request.method == 'POST':
        Cropname = request.form.get("Cropname")
        DistrictName = request.form.get("District")
        Production = float(request.form['Production'])
        Rainfall = float(request.form['Rainfall'])
        Area = float(request.form['Area'])

        features = np.array([[DistrictName, Cropname, Production, Rainfall, Area]], dtype=object)
        # Transform the features using the preprocessor
        transformed_features = preprocessor.transform(features)

        # Make the prediction
        predicted_yield = best_model.predict(transformed_features).reshape(1, -1)
        final_prediction = float(predicted_yield[0])
        return render_template('crop-result.html', prediction=final_prediction, title=title)

#render fertilizer recommendation result page

@ app.route('/fertilizer-recommend', methods=['POST'])
def fertilizer_recommend():
    title = 'Khetify - Fertilizer Suggestion'

    if request.method == 'POST':
        crop_name = str(request.form['cropname'])
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        # ph = float(request.form['ph'])

        df = pd.read_csv('data/fertilizer.csv')

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            elif n > 0:
                key = "Nlow"
            else:
                key = "Perfect"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            elif n > 0:
                key = "Plow"
            else:
                key = "Perfect"
        else:
            if k < 0:
                key = 'KHigh'
            elif n > 0:
                key = "Klow"
            else:
                key = "Perfect"

        response = Markup(str(fertilizer_dic[key]))

        return render_template('fertilizer-result.html', recommendation=response, title=title)


# render disease prediction result page
@app.route('/disease', methods=['GET','POST'])
def disease_prediction():
    title = 'Khetify - Disease Prediction'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html',title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return  render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


#-------------------------------------------#
if __name__ == '__main__':
    app.run(debug=False)