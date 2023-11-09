import numpy as np
import pandas as pd
import pickle
import bz2
from sklearn.preprocessing import StandardScaler
import gradio as gr
from firedanger import indices

#Standardization
R_pickle_in = bz2.BZ2File('regression.pkl', 'rb')
model_R = pickle.load(R_pickle_in)

df=pd.read_csv("./fwi_dataset_augmented.csv")
scaler=StandardScaler()
df.drop(['u10', 'v10','time','longitude','latitude','danger','year'], axis=1, inplace=True)
X = df.drop(['fwi'], axis=1)
X_reg_scaled = scaler.fit_transform(X.values)

def reg_to_class(i):
    
    pred_clf= ("low" if -1<i<5 else "moderate" if 5<i<10 else "high" if 10<i<20
              else "very high" if 20<i<30 else "extreme")
    
    return pred_clf  
	
	
#function for predict
def predict_fwi(wind,hum,temp,precip,month):

	dc=indices.dc(int(temp),int(precip),int(month),40.5,15)
	dmc=indices.dmc(int(temp),int(precip),int(hum),int(month),40.5,6)
	ffmc=indices.ffmc(int(temp),int(precip),int(wind),int(hum),85)
	isi=indices.isi(int(wind),float(ffmc))
	bui=indices.bui(int(dmc),float(dc))
	
	data=[month,precip,hum,temp,wind,dc,dmc,ffmc,isi,bui]
	data = [np.array(data)]
	data = scaler.transform(data)
	
	fwi=model_R.predict(data)[0]
	
	class_fwi=reg_to_class(fwi)
	
	return fwi,class_fwi
	
	
#App
title = "FWI prediction in Lecce"
description = "Please insert the required data."


input1 = gr.inputs.Number(label="Windspeed [m/s]")
input2 = gr.inputs.Number(label="Relative humidity at 12.00 [%]")
input3 = gr.inputs.Number(label="2m temperature at 12.00 [C]")
input4 = gr.inputs.Number(label="Total precipitation over the last 24h [mm]")
input5 = gr.inputs.Number(label="Month")

output1 = gr.inputs.Number(label='fwi')
output2= gr.Textbox(label="class")
    
    
gr.Interface(fn=predict_fwi,title=title,inputs=[input1,input2,input3,input4,input5],outputs=[output1,output2], 
			description=description).launch(
    debug=True)