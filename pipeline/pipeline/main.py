import dill

import pandas as pd 

from fastapi import FastAPI
from pydantic import BaseModel 

app = FastAPI()
model = dill.load(open('./models/user_action_predictor_1.pkl', mode = 'rb'))

class Form(BaseModel):
    hit_date : str
    hit_number : int
    hit_referer : str
    hit_page_path : str
    event_label : str
    visit_time : str
    visit_number : int
    utm_source : str
    utm_medium : str
    utm_campaign : str
    utm_adcontent : str
    device_category : str
    device_os : str
    device_screen_resolution : str
    device_browser : str
    geo_country : str
    geo_city : str

class Prediction(BaseModel):
    Result : int

@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict', response_model = Prediction)
def predict(form : Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'Result' : y[0],
    }