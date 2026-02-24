from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from PIL import Image
import base64
import io
import time
import pandas as pd
import os
import zipfile
import shutil
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import keras


app = Flask(__name__)

LAST_ACTIVITY = time.time()
TIMEOUT = 60 
UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "datasets"
TRAINED_MODEL_FOLDER = "trained_models"
PLOT_FOLDER = "static/plots"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(TRAINED_MODEL_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return render_template(
        'index.html',
        selected_model=selected_model,
        result=None,
        error="Ukuran file terlalu besar! Maksimal 5MB."
    ), 413

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

def update_activity():
    global LAST_ACTIVITY
    LAST_ACTIVITY = time.time()

MODEL_PATHS = {
    "VGG16": "model/VGG16.h5",
    "MobileNetV2": "model/MobileNetV2.h5"
}

selected_model = "VGG16"

loaded_models = {}

training_status = {
    "status": "idle",
    "progress": 0,
    "metrics": None,
    "model_path": None,
    "plots": {}
}

import keras

def get_model(model_name):

    if model_name not in loaded_models:

        print(f"[INFO] Loading model: {model_name}")

        loaded_models[model_name] = keras.models.load_model(
            MODEL_PATHS[model_name],
            compile=False
        )

        print("[INFO] Model loaded")

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
            return render_template(
                'index.html',
                selected_model=selected_model,
                result=None,
                error="No file selected"
            )

        if not file.filename.lower().endswith(('.png','.jpg','.jpeg')):
            return render_template(
                'index.html',
                selected_model=selected_model,
                result=None,
                error="Invalid file type"
            )

        try:

            result, threshold, img, analysis_time = predict_image(file.stream)

            image_data = None

            if result == "Bukan Judol":

                buffered = io.BytesIO()

                img.save(buffered, format="PNG")

                img_base64 = base64.b64encode(
                    buffered.getvalue()
                ).decode('utf-8')

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

            return render_template(
                'index.html',
                selected_model=selected_model,
                result=None,
                error=str(e)
            )

    return render_template(
        'index.html',
        selected_model=selected_model,
        result=None,
        error=None
    )

@app.route('/set_model', methods=['POST'])
def set_model():

    global selected_model

    model_name = request.form.get('model_name')

    if model_name in MODEL_PATHS:

        selected_model = model_name

        get_model(model_name)

        print(f"[INFO] Switched model")

    return redirect(url_for('index'))

@app.route('/compare')
def compare():

    csv_path = "hasil_eksperimen.csv"

    if not os.path.exists(csv_path):

        return render_template(
            "compare.html",
            columns=[],
            rows=[],
            best_rows={}
        )

    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')

    df.columns = df.columns.str.strip()

    best_rows = {}

    for model_name in df["Model"].unique():

        subset = df[df["Model"] == model_name]

        best_idx = subset["Akurasi Validasi"].idxmax()

        best_rows[best_idx] = model_name

    return render_template(
        "compare.html",
        columns=df.columns.tolist(),
        rows=df.values.tolist(),
        best_rows=best_rows
    )

@app.route("/train_page")
def train_page():
    return render_template("train.html")

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    
    update_activity()

    file = request.files["file"]

    zip_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(zip_path)

    extract_path = os.path.join(
        DATASET_FOLDER,
        "user_dataset"
    )

    if os.path.exists(extract_path):

        shutil.rmtree(extract_path)

    with zipfile.ZipFile(zip_path,'r') as zip_ref:

        zip_ref.extractall(extract_path)

    return jsonify({"status":"success"})

def build_model(model_name, lr, dropout, fine_tune=False):

    if model_name=="VGG16":

        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(224,224,3)
        )

    else:

        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224,224,3)
        )

    base_model.trainable=False

    if fine_tune:

        for layer in base_model.layers[-20:]:

            layer.trainable=True

    model = keras.Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(256,activation="relu"),

        layers.Dropout(dropout),

        layers.Dense(128,activation="relu"),

        layers.Dropout(dropout),

        layers.Dense(1,activation="sigmoid")

    ])

    model.compile(

        optimizer=keras.optimizers.Adam(lr),

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    return model

def train_model_thread(params):

    global training_status

    training_status["status"]="training"
    training_status["progress"]=0

    model_name=params["model"]
    lr=float(params["lr"])
    dropout=float(params["dropout"])
    epochs=int(params["epochs"])
    fine_tune=params["fine_tune"]

    train_dir="datasets/user_dataset/data/train"
    test_dir="datasets/user_dataset/data/test"

    img_size=(224,224)
    batch_size=32

    train_ds=tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    val_ds=tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    test_ds=tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="binary"
    )

    def normalize(x,y):
        return x/255.0,y

    train_ds=train_ds.map(normalize)
    val_ds=val_ds.map(normalize)
    test_ds=test_ds.map(normalize)

    model=build_model(
        model_name,
        lr,
        dropout,
        fine_tune
    )

    class ProgressCallback(keras.callbacks.Callback):

        def on_epoch_end(self,epoch,logs=None):

            training_status["progress"]=int(
                (epoch+1)/epochs*100
            )

    history=model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ProgressCallback()]
    )

    model_path=os.path.join(
        TRAINED_MODEL_FOLDER,
        f"{model_name}_trained.h5"
    )

    model.save(model_path)

    training_status["model_path"]=model_path

    # ambil label asli
    y_true = np.concatenate(
        [y.numpy() for x, y in test_ds],
        axis=0
    ).astype(int).flatten()

    # prediksi
    y_pred = model.predict(test_ds)
    y_pred = (y_pred > 0.5).astype(int).flatten()

    # classification report
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    cm_path = os.path.join(
        PLOT_FOLDER,
        "confusion.png"
    )

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(cm_path)
    plt.close()


    acc_path = os.path.join(
        PLOT_FOLDER,
        "accuracy.png"
    )

    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(["train", "validation"])
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(acc_path)
    plt.close()


    # cari label kelas otomatis
    class_keys = [
        k for k in report.keys()
        if k not in ("accuracy", "macro avg", "weighted avg")
    ]

    main_label = class_keys[-1] if class_keys else None

    training_status["metrics"] = {

        "accuracy": float(report.get("accuracy", 0)),

        "precision": float(
            report.get(main_label, {}).get("precision", 0)
        ),

        "recall": float(
            report.get(main_label, {}).get("recall", 0)
        ),

        "f1": float(
            report.get(main_label, {}).get("f1-score", 0)
        )

    }

    training_status["plots"] = {
        "confusion": cm_path,
        "accuracy": acc_path
    }

    training_status["progress"] = 100
    training_status["status"] = "finished"


@app.route("/train",methods=["POST"])
def train():
    update_activity()
    params=request.json

    thread=threading.Thread(
        target=train_model_thread,
        args=(params,)
    )

    thread.start()

    return jsonify({"status":"started"})


@app.route("/training_status")
def training_status_route():
    update_activity()
    return jsonify(training_status)


@app.route("/download_model")
def download_model():

    if training_status["model_path"]:

        return send_file(
            training_status["model_path"],
            as_attachment=True
        )

    return "No model"

def cleanup_files():

    folders = [
        DATASET_FOLDER,
        TRAINED_MODEL_FOLDER,
        PLOT_FOLDER,
        UPLOAD_FOLDER
    ]

    for folder in folders:

        if os.path.exists(folder):

            shutil.rmtree(folder)

            os.makedirs(folder, exist_ok=True)

    print("AUTO CLEANUP DONE")

def inactivity_monitor():

    global LAST_ACTIVITY

    while True:

        time.sleep(60)

        idle = time.time() - LAST_ACTIVITY

        if idle > TIMEOUT:

            cleanup_files()

            LAST_ACTIVITY = time.time()

if __name__ == '__main__':
    cleanup_files()
    threading.Thread(
        target=inactivity_monitor,
        daemon=True
    ).start()

    app.run(debug=True, use_reloader=False)