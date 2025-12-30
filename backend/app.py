from flask import Flask, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask_cors import CORS
import os

# ======== Load Model ========
model = tf.keras.models.load_model("models/efficientNet_model_tuned.h5")
IMG_SIZE = 300

app = Flask(__name__, template_folder="../templates")
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CORS(app)


# ======== Prediction Function ========
def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if pred_prob >= 0.5 else "NORMAL"
    return label, float(pred_prob)


# ======== Routes ========
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # save uploaded file
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)

            # predict
            label, confidence = predict_pneumonia(save_path)

            return render_template(
                "index.html",
                result=label,
                confidence=f"{confidence * 100:.2f}",
                file_path=f"/uploads/{file.filename}"
            )

    return render_template("index.html", result=None)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
