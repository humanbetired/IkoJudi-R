from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import io
import time
import pandas as pd
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  

MODEL_PATHS = {
    "VGG16": "model/VGG16.h5",
    "MobileNetV2": "model/MobileNetV2.h5"
}

selected_model = "VGG16"

loaded_models = {}

def get_model(model_name):
    if model_name not in loaded_models:
        print(f"[INFO] Loading model: {model_name}")
        loaded_models[model_name] = load_model(MODEL_PATHS[model_name], compile=False)
        print(f"[INFO] Model {model_name} loaded")
    return loaded_models[model_name]

model = get_model(selected_model)

print("[INFO] Server ready")

def predict_image(file_stream):

    start_time = time.time()

    img = Image.open(file_stream).convert('RGB')
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    model = get_model(selected_model)

    prediction = model.predict(img_array, verbose=0)[0][0]

    analysis_time = time.time() - start_time

    label = "Judol" if prediction <= 0.5 else "Bukan Judol"

    return label, float(prediction), img, analysis_time

@app.route('/', methods=['GET', 'POST'])
def index():

    global selected_model

    if request.method == 'POST':

        file = request.files.get('image')

        if not file:
            return render_template('index.html',
                                   selected_model=selected_model,
                                   result=None,
                                   error="No file selected")

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('index.html',
                                   selected_model=selected_model,
                                   result=None,
                                   error="Invalid file type")

        try:

            result, threshold, img, analysis_time = predict_image(file.stream)

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")

            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data = f"data:image/png;base64,{img_base64}"

            return render_template(
                'index.html',
                selected_model=selected_model,
                result=result,
                threshold=threshold,
                image_data=image_data,
                analysis_time=analysis_time,
                error=None
            )

        except Exception as e:

            return render_template('index.html',
                                   selected_model=selected_model,
                                   result=None,
                                   error=str(e))

    return render_template('index.html',
                           selected_model=selected_model,
                           result=None,
                           error=None)


@app.route('/set_model', methods=['POST'])
def set_model():

    global selected_model

    model_name = request.form.get('model_name')

    if model_name in MODEL_PATHS:

        selected_model = model_name

        get_model(model_name)

        print(f"[INFO] Switched to model: {model_name}")

    return redirect(url_for('index'))


@app.route('/compare')
def compare():

    csv_path = "hasil_eksperimen.csv"

    if not os.path.exists(csv_path):
        return render_template("compare.html",
                               columns=[],
                               rows=[],
                               best_rows=[])

    df = pd.read_csv(csv_path)

    best_rows = {}

    for model_name in df["Model"].unique():

        subset = df[df["Model"] == model_name]
        best_idx = subset["Akurasi Validasi"].idxmax()

        best_rows[best_idx] = model_name

    return render_template("compare.html",
                           columns=df.columns.tolist(),
                           rows=df.values.tolist(),
                           best_rows=best_rows)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)