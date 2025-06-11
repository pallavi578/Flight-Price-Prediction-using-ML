from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model (make sure the model is saved as 'flight_model.pkl')
model = joblib.load('flight_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the form data
        departure_time = 1 if request.form.get('departure_time') == 'Afternoon' else 0
        arrival_time = 1 if request.form.get('arrival_time') == 'Afternoon' else 0
        source_place = request.form.get('source_place')
        destination_place = request.form.get('destination_place')
        number_of_stops = int(request.form.get('number_of_stops'))
        airline_type = request.form.get('airline_type')

        # Encode categorical inputs like source, destination, and airline type
        source_place_encoded = {'New York': 0, 'Los Angeles': 1, 'Chicago': 2, 'San Francisco': 3}.get(source_place)
        destination_place_encoded = {'London': 0, 'Paris': 1, 'Tokyo': 2, 'Sydney': 3}.get(destination_place)
        airline_type_encoded = {'Economy': 0, 'Business': 1, 'First Class': 2}.get(airline_type)

        # Prepare the input data as a pandas DataFrame
        input_data = pd.DataFrame([[departure_time, arrival_time, source_place_encoded,
                                    destination_place_encoded, number_of_stops, airline_type_encoded]],
                                  columns=['departure_time', 'arrival_time', 'source_place', 
                                           'destination_place', 'number_of_stops', 'airline_type'])

        # Make the prediction
        prediction = model.predict(input_data)[0]
        prediction_result = "High priority" if prediction == 1 else "Low priority"

        return render_template('index.html', prediction_result=prediction_result, 
                               departure_time=departure_time, arrival_time=arrival_time,
                               source_place=source_place, destination_place=destination_place,
                               number_of_stops=number_of_stops, airline_type=airline_type)

    return render_template('index.html', prediction_result=prediction)

if __name__ == '__main__':
    app.run(debug=True)