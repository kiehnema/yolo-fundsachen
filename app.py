import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np

st.set_page_config(page_title="KI Objekterkennung (YOLO‑World)", layout="centered")

st.title("🧠 YOLO‑World Objekterkennung")
st.write("Bild hochladen und Objekte via Text‑Prompt erkennen.")

# Modell laden
@st.cache_resource
def load_yoloworld():
    return YOLOWorld("yolov8s-world.pt")

model = load_yoloworld()

# Eingabefeld für Objektnamen (Deutsch/Englisch möglich)
classes_input = st.text_input(
    "Welche Objekte soll die KI erkennen? (Komma getrennt)",
    "hat, key, phone, wallet, backpack"
)

prompt_list = [c.strip() for c in classes_input.split(",") if c.strip()]

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg","jpeg","png"]
)

if uploaded_file and prompt_list:

    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    st.write("🔍 KI analysiert mit YOLO‑World…")

    # Bild für Berechnung in numpy
    img_array = np.array(image)

    # Prompt setzen (Text Klassen)
    model.set_classes(prompt_list)

    # Vorhersage
    results = model.predict(img_array)

    # Annotiertes Bild anzeigen
    annotated = results[0].plot()
    st.image(annotated, caption="Erkannte Objekte", use_column_width=True)

    # Anzeige der gefundenen Objekte
    labels = results[0].boxes.cls
    detected = [prompt_list[int(idx)] for idx in labels] if len(labels)>0 else []
    if detected:
        st.success("Gefunden: " + ", ".join(set(detected)))
    else:
        st.warning("Keine der eingegebenen Objekte gefunden.")
