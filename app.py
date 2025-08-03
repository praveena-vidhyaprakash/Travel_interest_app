from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('travel_model.pkl')
destination_map = joblib.load('destination_map.pkl')
travel_type_map = joblib.load('travel_type_map.pkl')

destination_map_rev = {v: k for k, v in destination_map.items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        travel_type = request.form['travel_type']
        travel_encoded = travel_type_map.get(travel_type, 0)
        
        pred_num = model.predict([[age, gender, travel_encoded]])[0]
        prediction = destination_map_rev.get(pred_num, "Unknown")
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)