import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Digitales KI-Fundbüro", layout="wide")

UPLOAD_FOLDER = "uploads"
DATA_FILE = "fundliste.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("🧠 Digitales KI-Fundbüro")
st.write("Lade ein Bild hoch – die KI erkennt automatisch Objekte und speichert den Fund.")

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLOWorld("yolov8s-world.pt")

model = load_model()

# -----------------------------
# DATA HANDLING
# -----------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["zeit", "datei", "objekte", "fundort", "beschreibung"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

df = load_data()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # KI Analyse
    with st.spinner("🔍 KI analysiert Bild..."):
        results = model.predict(np.array(image))
        annotated = results[0].plot()

    st.image(annotated, caption="Erkannte Objekte", use_column_width=True)

    labels = results[0].boxes.cls
    names = results[0].names

    if len(labels) > 0:
        detected_objects = list(set([names[int(i)] for i in labels]))
        detected_str = ", ".join(detected_objects)

        st.success(f"Gefunden: {detected_str}")

        # Prüfen ob schon vorhanden
        exists = (
            (df["datei"] == uploaded_file.name) &
            (df["objekte"] == detected_str)
        ).any()

        if not exists:
            # Bild speichern
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            if not os.path.exists(image_path):
                image.save(image_path)

            # Neuer Eintrag
            new_entry = pd.DataFrame([{
                "zeit": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "datei": uploaded_file.name,
                "objekte": detected_str,
                "fundort": "",
                "beschreibung": ""
            }])

            df = pd.concat([df, new_entry], ignore_index=True)
            save_data(df)

            st.success("✅ Fund gespeichert!")
        else:
            st.info("⚠️ Dieser Fund existiert bereits.")
    else:
        st.warning("Keine Objekte erkannt.")

# -----------------------------
# FUNDE ANZEIGEN
# -----------------------------
st.divider()
st.header("📋 Gespeicherte Funde")

if len(df) == 0:
    st.info("Noch keine Funde vorhanden.")
else:
    for i, row in df.iterrows():
        with st.expander(f"Fund vom {row['zeit']}"):

            col1, col2 = st.columns([1, 2])

            # Bild anzeigen
            with col1:
                img_path = os.path.join(UPLOAD_FOLDER, row["datei"])
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)

            # Infos + Bearbeitung
            with col2:
                st.write(f"**Objekte:** {row['objekte']}")

                new_location = st.text_input(
                    "📍 Fundort",
                    value=row["fundort"],
                    key=f"loc_{i}"
                )

                new_desc = st.text_area(
                    "📝 Beschreibung",
                    value=row["beschreibung"],
                    key=f"desc_{i}"
                )

                if st.button("💾 Speichern", key=f"save_{i}"):
                    df.at[i, "fundort"] = new_location
                    df.at[i, "beschreibung"] = new_desc
                    save_data(df)
                    st.success("Gespeichert!")

                if st.button("🗑️ Löschen", key=f"delete_{i}"):
                    df = df.drop(i).reset_index(drop=True)
                    save_data(df)
                    st.warning("Eintrag gelöscht!")
                    st.rerun()
