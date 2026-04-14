"""
Streamlit app for Hugging Face Spaces: model + metadata from S3, same logic as Src/predict.py.
Run from repo root: streamlit run App/app.py
CHECKPOINT ARCHITECTURE: 25088→4096→1024→nb_classes
"""
import sys
from pathlib import Path

# Repo root on Hugging Face
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os

import boto3
import pandas as pd
import streamlit as st
import torch
import torchvision.models as models
from PIL import Image

from Src.preprocessing import preprocess_image

# Config secrets
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY")
S3_CSV_KEY = os.getenv("S3_CSV_KEY", "Data/mushroom2.csv")

# Writable cache on HF Spaces containers
MODEL_CACHE_PATH = os.getenv("MODEL_CACHE_PATH", "/tmp/model.pth")


def _build_idx2info(classes: list, desc_map: dict) -> dict:
    """Same mapping as Src/predict.py."""
    return {
        i: {
            "scientific_name": name,
            "description": desc_map.get(name, {}).get("description", ""),
            "edibility": desc_map.get(name, {}).get("edibility", ""),
        }
        for i, name in enumerate(classes)
    }


@st.cache_resource(show_spinner="Chargement du modèle depuis S3...")
def load_model_from_s3():
    """
    Download checkpoint + CSV from S3, build VGG19 with custom classifier.
    Checkpoint architecture: 25088→4096→1024→nb_classes
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    if not os.path.exists(MODEL_CACHE_PATH):
        s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, MODEL_CACHE_PATH)

    checkpoint = torch.load(MODEL_CACHE_PATH, map_location=torch.device("cpu"))

    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=S3_CSV_KEY)
    df = pd.read_csv(obj["Body"])
    df = df[["scientific_name", "new_description", "edibility"]].dropna().drop_duplicates()

    desc_map = {
        row["scientific_name"]: {
            "description": row["new_description"],
            "edibility": row["edibility"],
        }
        for _, row in df.iterrows()
    }

    if isinstance(checkpoint, dict) and "classes" in checkpoint:
        classes = checkpoint["classes"]
    elif isinstance(checkpoint, dict) and "idx2label" in checkpoint:
        idx2label = checkpoint["idx2label"]
        classes = [idx2label[i] for i in range(len(idx2label))]
    else:
        classes = sorted(df["scientific_name"].unique().tolist())

    idx2info = _build_idx2info(classes, desc_map)
    nb_classes = len(classes)

    # Build model with CUSTOM classifier: 25088→4096→1024→nb_classes
    model = models.vgg19(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),      # First layer: 4096 units
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4096, 1024),       # Second layer: 1024 units (not 4096!)
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, nb_classes), # Output layer
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model, idx2info


def predict_top_k(model, idx2info: dict, image: Image.Image, top_k: int = 3):
    tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze().numpy()

    top_indices = probs.argsort()[::-1][:top_k]
    return [
        {
            "idx": int(i),
            "name": idx2info[i]["scientific_name"],
            "description": idx2info[i]["description"] or "Description non disponible",
            "edibility": idx2info[i]["edibility"] or "Toxicity non disponible",
            "confidence": float(probs[i]),
        }
        for i in top_indices
    ]


# ui

st.set_page_config(
    page_title="Identifier un champignon",
    page_icon="🍄",
    layout="centered",
)

st.title("Identifier un champignon 🍄")
st.caption("Modèle : VGG19 fine-tuné · Poids & données : S3 · Pipeline : Src/")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME, S3_MODEL_KEY]):
    st.error(
        "Variables d'environnement AWS manquantes. "
        "Définissez AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, "
        "S3_BUCKET_NAME et S3_MODEL_KEY dans Hugging Face Spaces > Settings > Secrets."
    )
    st.stop()

try:
    model, idx2info = load_model_from_s3()
    st.sidebar.success(f"Modèle chargé — {len(idx2info)} espèces")
except Exception as e:
    st.error(f"Impossible de charger le modèle depuis S3 : {e}")
    st.stop()

st.write("Veuillez uploader une photo de champignon (PNG, JPG, etc.) :")
uploaded_file = st.file_uploader(
    "",
    type=["png", "jpg", "jpeg", "gif", "bmp"],
    label_visibility="collapsed",
    accept_multiple_files=False,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploadée", use_column_width=True)

    with st.spinner("Analyse en cours..."):
        try:
            results = predict_top_k(model, idx2info, image, top_k=3)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.stop()

    if results:
        top = results[0]

        def circular_progress(value):
            percent = int(value * 100)
            color = "#4CAF50" if percent > 80 else "#FF9800" if percent > 40 else "#F44336"
            st.markdown(
                f"""
            <div style="
                position: relative;
                width: 100px;
                height: 100px;
                border-radius: 50%;
                background: conic-gradient({color} {percent}%, #e0e0e0 {percent}%);
                display: flex;
                align-items: center;
                justify-content: center;
                margin: auto;
            ">
                <div style="
                    width: 70px;
                    height: 70px;
                    background: rgb(14, 17, 23);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 20px;
                    font-weight: bold;
                    color: #fff;
                ">
                    {percent}%
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"## {top['name']} — {top['edibility'].capitalize()}")
            st.markdown(
                f"<p style='font-size:18px;'>{top['description']}</p>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
            ">
            """,
                unsafe_allow_html=True,
            )
            circular_progress(top["confidence"])
            st.markdown("</div>", unsafe_allow_html=True)

        if len(results) > 1:
            st.markdown("---")
            st.write("**Autres hypothèses :**")
            for rank, r in enumerate(results[1:], start=2):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {rank}. {r['name']} — {r['edibility'].capitalize()}")
                    st.markdown(
                        f"<p style='font-size:16px;'>{r['description']}</p>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        """
                    <div style="
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                    """,
                        unsafe_allow_html=True,
                    )
                    circular_progress(r["confidence"])
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")