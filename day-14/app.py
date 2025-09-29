import tensorflow as tf
from flask import Flask, request, jsonify

model = tf.keras.models.load_model("day-14/iris_model.keras")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']   # e.g. 
    prediction = model.predict([features])
    predicted_class = int(prediction.argmax(axis=1)[0])
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)