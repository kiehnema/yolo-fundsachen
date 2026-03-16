import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import os
from datetime import datetime

# Modell laden
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "uploads"
DATA_FILE = "fundliste.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("Digitales Fundbüro mit KI")

st.write("Lade ein Bild eines gefundenen Gegenstands hoch.")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild speichern
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    image.save(filepath)

    # KI Analyse
    results = model(filepath)

    labels = []
    for r in results:
        for c in r.boxes.cls:
            labels.append(model.names[int(c)])

    if labels:
        detected = ", ".join(set(labels))
    else:
        detected = "Unbekannt"

    st.success(f"Erkannte Objekte: {detected}")

    # Fund speichern
    entry = {
        "zeit": datetime.now(),
        "datei": uploaded_file.name,
        "erkannte_objekte": detected
    }

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(DATA_FILE, index=False)

    st.success("Fund wurde gespeichert!")

# Fundliste anzeigen
st.header("Gespeicherte Funde")

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    st.dataframe(df)
else:
    st.write("Noch keine Einträge vorhanden.")
