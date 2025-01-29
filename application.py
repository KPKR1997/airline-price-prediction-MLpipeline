import logging
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            app.logger.debug("Form data received:")
            app.logger.debug(request.form)

            data = CustomData(
                airline=request.form.get('airline'),
                source_city=request.form.get('source_city'),
                departure_time=request.form.get('departure_time'),
                stops=request.form.get('stops'),
                arrival_time=request.form.get('arrival_time'),
                destination_city=request.form.get('destination_city'),
                ticketclass=request.form.get('ticketclass'),
                duration=float(request.form.get('duration')),
                days_left=float(request.form.get('days_left'))
            )

            pred_df = data.get_data_as_data_frame()
            app.logger.debug("DataFrame created:")
            app.logger.debug(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            app.logger.debug(f"Prediction results: {results}")

            return render_template('index.html', results=results[0])
        except Exception as e:
            app.logger.error(f"Error occurred: {e}", exc_info=True)
            return render_template('index.html', error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)