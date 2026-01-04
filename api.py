from fastapi import FastAPI
from fastapi.responses import JSONResponse,HTMLResponse
from pydantic import BaseModel,Field
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware 

model=joblib.load('WineQualityModel.pkl')

class WineData(BaseModel):
	fixed_acidity:float
	volatile_acidity:float
	citric_acid:float
	residual_sugar:float
	chlorides:float
	free_sulfur_dioxide:float
	total_sulfur_dioxide:float
	density:float
	pH:float
	sulphates:float
	alcohol:float
	
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/',response_class=HTMLResponse)
def home():
	with open('index.html','r')as f:
	    return f.read()
	
@app.post('/predict')
def predict_output(d:WineData):
	df = pd.DataFrame([d.dict()])
	result=model.predict(df)[0]
	if result==0:
		output='Bad Quality'
	elif result==1:
		output='Good Quality'
	elif result==2:
		output='Execelent Quality'
	else:
		output='Wrong With input'
	return JSONResponse({'output':output})

	