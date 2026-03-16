import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="Digitales KI-Fundbüro", layout="wide")
st.title("🧠 Digitales KI-Fundbüro")
st.write("Bilder hochladen, Objekte automatisch erkennen, Fundliste anzeigen & filtern.")

# Ordner & CSV-Datei
UPLOAD_FOLDER = "uploads"
DATA_FILE = "fundliste.csv"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Modell laden
# -----------------------------
@st.cache_resource
def load_model():
    return YOLOWorld("yolov8s-world.pt")  # Zero-shot Modell

model = load_model()

# -----------------------------
# Eingaben
# -----------------------------
classes_input = st.text_input(
    "Welche Objekte soll die KI erkennen? (Komma getrennt)",
    "hat, key, wallet, phone, backpack"
)
prompt_list = [c.strip() for c in classes_input.split(",") if c.strip()]

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg","jpeg","png"]
)

# -----------------------------
# KI Analyse & Fund speichern
# -----------------------------
if uploaded_file and prompt_list:

    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    model.set_classes(prompt_list)
    img_array = np.array(image)
    st.write("🔍 KI analysiert das Bild…")
    results = model.predict(img_array)

    annotated = results[0].plot()
    st.image(annotated, caption="Erkannte Objekte", use_column_width=True)

    labels = results[0].boxes.cls
    detected = [prompt_list[int(idx)] for idx in labels] if len(labels) > 0 else []

    if detected:
        st.success("Gefunden: " + ", ".join(set(detected)))

        entry = {
            "zeit": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fundort": "",  # leer, kann später ergänzt werden
            "beschreibung": "",
            "datei": uploaded_file.name,
            "erkannte_objekte": ", ".join(set(detected))
        }

        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        else:
            df = pd.DataFrame([entry])

        df.to_csv(DATA_FILE, index=False)
        image.save(os.path.join(UPLOAD_FOLDER, uploaded_file.name))

        st.success("Fund wurde gespeichert!")
    else:
        st.warning("Keine der eingegebenen Objekte erkannt. Fund nicht gespeichert.")

# -----------------------------
# Fundliste anzeigen & filtern
# -----------------------------
st.header("Alle Funde")

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)

    # Filter
    filter_object = st.multiselect(
        "Nach Objekt filtern:",
        options=list(set(sum([s.split(", ") for s in df["erkannte_objekte"].tolist()], [])))
    )
    filter_location = st.text_input("Nach Fundort filtern:")

    df_filtered = df.copy()
    if filter_object:
        df_filtered = df_filtered[df_filtered["erkannte_objekte"].apply(
            lambda x: any(o in x for o in filter_object)
        )]
    if filter_location:
        df_filtered = df_filtered[df_filtered["fundort"].str.contains(filter_location, case=False)]

    st.dataframe(df_filtered)
else:
    st.write("Noch keine Einträge vorhanden.")
