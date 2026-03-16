import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Seiteneinstellungen
st.set_page_config(
    page_title="KI Objekterkennung",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 KI Objekterkennung")
st.write("Lade ein Bild hoch und die KI erkennt automatisch die Objekte darauf.")

# Modell laden (wird einmal geladen und dann gecached)
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# Datei Upload
uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Bild öffnen
    image = Image.open(uploaded_file)

    st.subheader("Hochgeladenes Bild")
    st.image(image, use_column_width=True)

    # Bild in numpy umwandeln
    img_array = np.array(image)

    st.write("🔍 KI analysiert das Bild...")

    # Objekterkennung
    results = model(img_array)

    # Annotiertes Bild mit Bounding Boxes
    annotated_image = results[0].plot()

    st.subheader("Erkannte Objekte")
    st.image(annotated_image, use_column_width=True)

    # Erkannte Klassen sammeln
    labels = []
    for cls in results[0].boxes.cls:
        labels.append(model.names[int(cls)])

    # Ergebnisse anzeigen
    if labels:
        unique_labels = list(set(labels))
        st.success("Erkannt wurden:")
        for obj in unique_labels:
            st.write(f"• {obj}")
    else:
        st.warning("Keine Objekte erkannt.")

st.markdown("---")
st.caption("KI basiert auf YOLO Objekt-Erkennung.")
