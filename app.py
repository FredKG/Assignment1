from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# load trained model
model = pickle.load(open("car_price_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    prediction = model.predict(final_features)

    output = round(float(prediction.flatten()[0]), 2)

    return render_template(
        "index.html",
        prediction_text="Predicted Car Price: {}".format(output)
    )

if __name__ == "__main__":
    app.run(debug=True)
