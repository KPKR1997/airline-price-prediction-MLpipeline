from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app=application

@app.route('/', methods = ['GET', 'POST'])

def predict_datapoint():
    if request.method =='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            airline = request.form.get('airline'),

            source_city = request.form.get('source_city'),
            
            departure_time = request.form.get('departure_time'),
            
            stops = request.form.get('stops'),
            
            arrival_time = request.form.get('arrival_time'),
            
            destination_city = request.form.get('destination_city'),
            
            ticketclass = request.form.get('ticketclass'),
            
            duration = float(request.form.get('duration')),
            days_left = float(request.form.get('days_left'))
        )

        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results =  results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug =True)