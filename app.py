from flask import Flask, render_template, request, redirect
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

# render Homepage
@app.route('/')
def home():
    title = 'Khetify - Home'
    return render_template('index.html',title=title)

#render crop yield prediction from page
@ app.route('/crop')
def crop_predict():
    title = 'Khetify - Crop Yield Prediction'
    return render_template('crop.html', title=title)

#render fertilizer recommendation form page
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = ' Khelify - Fertilizer Recommendation'

    return render_template('fertilizer.html',title=title)

#render about page
@app.route('/about')
def about():
    return render_template('about.html')

#render aboutcrop
@app.route('/about_crop')
def about_crop():
    return render_template('about-crop.html')

#render aboutfer
@app.route('/about_fer')
def about_fer():
    return render_template('about-fer.html')

#render aboutplant
@app.route('/about_plant')
def about_plant():
    return render_template('about-plant.html')


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
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

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
