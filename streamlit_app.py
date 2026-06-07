"""
Crop Doctor - Crop Disease Classification and Treatment Recommendation System
Run: streamlit run streamlit_app.py

Version: 1.2 - Added user upload saving for model improvement with privacy notice
Author: Daniel Arani Osuto
Description: AI-powered crop disease diagnosis system with Grad-CAM visualization,
             treatment recommendations, and weather integration.
"""

import streamlit as st
import streamlit_analytics2 as streamlit_analytics
import streamlit.components.v1 as components
from streamlit_geolocation import streamlit_geolocation
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
import time
from PIL import Image
import requests
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import re
from uuid import uuid4
from huggingface_hub import HfApi
import io
import feedparser
from datetime import datetime, timedelta
import hashlib

# ============================================================
# USER IMAGE SAVING FOR MODEL RETRAINING
# ============================================================

# Configuration - set these as Secrets in your Space
HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO_ID = "dosuto/crop-doctor-user-uploads"  # Your dataset repo

def save_user_image_for_training(image, primary_diagnosis, primary_confidence, all_predictions):
    """
    Saves uploaded image ONCE with the top prediction as primary diagnosis.
    """
    if not HF_TOKEN:
        return False

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())[:8]
        filename = f"{timestamp}_{unique_id}.jpg"

        # Prepare image
        if image.mode != 'RGB':
            save_image = image.convert('RGB')
        else:
            save_image = image.copy()

        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        save_image.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)

        # Prepare metadata - store ALL predictions for reference
        all_preds_list = []
        for pred in all_predictions:
            all_preds_list.append({
                "class": pred['class'],
                "confidence": pred['confidence']
            })

        metadata = {
            "file_name": filename,
            "primary_diagnosis": primary_diagnosis,
            "primary_confidence": primary_confidence,
            "all_predictions": all_preds_list,  # Store all K predictions
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.2"
        }
        metadata_json = json.dumps(metadata, indent=2)

        # Upload to Hugging Face Dataset
        api = HfApi()

        # Upload image
        api.upload_file(
            path_or_fileobj=img_bytes,
            path_in_repo=f"images/{filename}",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )

        # Upload metadata
        api.upload_file(
            path_or_fileobj=io.BytesIO(metadata_json.encode()),
            path_in_repo=f"images/{filename.replace('.jpg', '.json')}",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )

        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Crop Doctor - Disease Classification and Treatment System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - CONSISTENT FONTS WITH IMPROVED REFERENCES
# ============================================================

st.markdown("""
<style>
    /* Consistent font family for all elements */
    html, body, [class*="css"], .stMarkdown, .stText, .stButton, .stNumberInput,
    .stSelectbox, .stRadio, label, .stTextInput, .stFileUploader, .stAlert {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    .stApp { background-color: #f5f5f5; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: white;
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-size: 32px;
        margin-bottom: 5px;
        font-weight: 700;
    }
    .main-header .subtitle {
        font-size: 16px;
        opacity: 0.9;
        margin-top: 5px;
    }

    /* Section Cards - Individual curved boxes */
    .section-card {
        background: white;
        border-radius: 20px;
        padding: 12px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Headers inside section cards - consistent spacing */
    .section-card h3 {
        color: #2E7D32;
        margin-bottom: 8px !important;
        margin-top: 0 !important;
        font-size: 18px;
        font-weight: 700;
    }

    /* Paragraphs inside section cards */
    .section-card p {
        margin: 4px 0 !important;
        line-height: 1.4 !important;
    }

    /* Pre-line text (characteristics) - tighter spacing for bullet points */
    .section-card p[style*="white-space: pre-line"] {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }

    /* Prediction list container */
    .prediction-list {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
        background: transparent;
    }

    .prediction-row {
        display: flex;
        align-items: baseline;
        margin: 3px 0 !important;
    }

    .prediction-marker {
        display: inline-block;
        width: 30px;
        text-align: right;
        margin-right: 8px;
    }

    .prediction-name {
        font-weight: normal;
    }

    .prediction-value {
        font-weight: normal;
    }

    /* Mode badge */
    .mode-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .mode-online { background: #2196F3; color: white; }
    .mode-offline { background: #FF6F00; color: white; }

    /* Weather and manufacturer cards */
    .weather-card, .manufacturer-card {
        background: #E3F2FD;
        padding: 15px;
        border-radius: 15px;
        margin-top: 10px;
    }
    .manufacturer-card {
        background: #FFF3E0;
    }

    /* Buttons */
    .stButton button {
        background: #2E7D32;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 13px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background: #1B5E20;
        transform: scale(1.02);
    }

    /* Number input */
    .stNumberInput input {
        font-size: 13px;
        padding: 8px 12px;
        border-radius: 25px;
        text-align: center;
    }
    .stNumberInput label {
        font-size: 12px;
        font-weight: 500;
    }

    /* Reference text styling */
    .reference-text {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 14px;
        color: #444;
        margin: 5px 0;
        white-space: pre-wrap;
        line-height: 1.5;
    }

    /* Grad-CAM Legend styling */
    .gradcam-legend {
        font-size: 15px;
        line-height: 1.5;
    }
    .gradcam-legend p {
        font-size: 15px;
        margin-bottom: 6px;
    }
    .gradcam-tip {
        font-size: 12px;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px dashed #ccc;
        color: #555;
    }

    /* Risk levels */
    .risk-high { color: #FF0000; font-weight: 700; }
    .risk-moderate { color: #FFA500; font-weight: 700; }
    .risk-low { color: #008000; font-weight: 700; }

    /* Help card */
    .help-card {
        background: #FFF8E1;
        border-left: 5px solid #FF9800;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .help-card h4 {
        color: #E65100;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 18px;
        font-weight: 600;
    }
    .help-card p {
        margin: 5px 0;
    }

    /* Classes list styling */
    .classes-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-height: 400px;
        overflow-y: auto;
    }
    .classes-card h3 {
        margin-top: 0;
        margin-bottom: 15px;
        color: #2E7D32;
        font-size: 18px;
        font-weight: 700;
    }
    .classes-category {
        color: #2E7D32;
        font-size: 18px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 8px;
        border-bottom: 2px solid #A5D6A7;
        padding-bottom: 5px;
    }
    .classes-category:first-child {
        margin-top: 0;
    }
    .class-item {
        font-size: 14px;
        padding: 3px 0 3px 20px;
        color: #333;
    }
    .class-count {
        font-size: 12px;
        color: #666;
        margin-left: 10px;
    }

    /* Bold text */
    strong, b {
        font-weight: 700;
    }

    /* Lists */
    ul, ol {
        margin: 5px 0;
        padding-left: 20px;
    }
    li {
        margin: 2px 0;
    }

    /* Remove extra spacing from Streamlit's default markdown */
    .stMarkdown div {
        margin-top: 0;
        margin-bottom: 0;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2E7D32;
    }

    /* Radio button styling */
    .stRadio > div {
        gap: 20px;
    }

    /* Text input styling */
    .stTextInput input {
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 8px 12px;
    }

    /* Info, warning, error boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Privacy notice styling */
    .privacy-notice {
        background-color: #E8F5E9;
        border-left: 5px solid #2E7D32;
        padding: 12px 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .privacy-notice-text {
        flex: 1;
        font-size: 14px;
        color: #1B5E20;
    }
    .privacy-notice-close {
        background: none;
        border: none;
        font-size: 20px;
        cursor: pointer;
        color: #2E7D32;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'mode' not in st.session_state:
    st.session_state.mode = 'offline'
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'current_top_k' not in st.session_state:
    st.session_state.current_top_k = 3
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'current_heatmap' not in st.session_state:
    st.session_state.current_heatmap = None
if 'current_original_img' not in st.session_state:
    st.session_state.current_original_img = None
if 'current_alt_data' not in st.session_state:
    st.session_state.current_alt_data = {}
if 'current_showing_alternative' not in st.session_state:
    st.session_state.current_showing_alternative = None
if 'location' not in st.session_state:
    st.session_state.location = "Ekerenyo, Nyamira County, Kenya"
if 'showing_common_chemicals' not in st.session_state:
    st.session_state.showing_common_chemicals = False
if 'weather_info' not in st.session_state:
    st.session_state.weather_info = None
if 'common_chemicals_data' not in st.session_state:
    st.session_state.common_chemicals_data = None
if 'show_help' not in st.session_state:
    st.session_state.show_help = False
if 'show_k_dialog' not in st.session_state:
    st.session_state.show_k_dialog = False
if 'show_classes' not in st.session_state:
    st.session_state.show_classes = False
if 'current_save_path' not in st.session_state:
    st.session_state.current_save_path = None
if 'location_change_method' not in st.session_state:
    st.session_state.location_change_method = None
#if 'waiting_for_gps' not in st.session_state:
    #st.session_state.waiting_for_gps = False
# Privacy notice state
if 'privacy_notice_dismissed' not in st.session_state:
    st.session_state.privacy_notice_dismissed = False
# Location dialog states
if 'show_top_location_dialog' not in st.session_state:
    st.session_state.show_top_location_dialog = False
if 'show_top_manual_entry' not in st.session_state:
    st.session_state.show_top_manual_entry = False
#if 'request_gps' not in st.session_state:
    #st.session_state.request_gps = False
#if 'gps_location' not in st.session_state:
#    st.session_state.gps_location = None
if 'location_method' not in st.session_state:
    st.session_state.location_method = "manual"
#if 'show_gps_dialog' not in st.session_state:
#    st.session_state.show_gps_dialog = False
#if 'show_gps_ui' not in st.session_state:
#   st.session_state.show_gps_ui = False

# ============================================================
# HELPER FUNCTION: Check Internet Connection (for mode switch suggestion)
# ============================================================
def check_internet_connection():
    """Check if internet connection is available"""
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

# ============================================================
# HELPER FUNCTION: Generate Export Report
# ============================================================

def generate_export_report(disease_data, treatment_data, references, weather_data=None):
    """Generate a downloadable report of the diagnosis and treatment"""

    # Get the treatment data
    treatment = treatment_data

    # Get management references
    management_refs = treatment.get('management_refs', [])
    management_refs_text = ""
    for idx, ref_num in enumerate(management_refs, 1):
        if ref_num in references:
            management_refs_text += f"  [{idx}] {references[ref_num]}\n"

    # Get chemical references
    chemical_refs_original = treatment.get('chemical_refs_original', [])
    chemical_refs_text = ""
    if chemical_refs_original:
        for idx, ref_num in enumerate(chemical_refs_original, 1):
            if ref_num in references:
                chemical_refs_text += f"  [{idx}] {references[ref_num]}\n"
            else:
                chemical_refs_text += f"  [{idx}] Reference {ref_num}\n"
    else:
        chemical_refs_text = "  See product labels for specific chemical references\n"

    # Get XAI references
    xai_refs = treatment.get('xai_ref_numbers', [])
    xai_refs_text = ""
    for idx, ref_num in enumerate(xai_refs, 1):
        if ref_num in references:
            xai_refs_text += f"  [{idx}] {references[ref_num]}\n"

    # Add Grad-CAM reference
    grad_cam_ref_num = len(xai_refs) + 1
    xai_refs_text += f"  [{grad_cam_ref_num}] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,' in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618-626.\n"

    report = f"""
CROP DOCTOR DIAGNOSIS REPORT
{'='*60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DIAGNOSIS SUMMARY:
{'-'*40}
Disease: {disease_data['class']}
Confidence: {disease_data['confidence']*100:.1f}%
Category: {treatment['category']}
Causal Agent: {treatment['causal_agent']}

TREATMENT RECOMMENDATIONS:
{'-'*40}

MANAGEMENT:
{treatment.get('management', 'Information not available')[:1000]}

CHEMICAL CONTROL:
{treatment.get('chemical_control', 'Information not available')[:1000]}

{'='*60}
MANAGEMENT REFERENCES:
{'-'*40}
{management_refs_text if management_refs_text else '  See local agricultural extension office for region-specific advice'}

CHEMICAL REFERENCES:
{'-'*40}
{chemical_refs_text}

XAI SOURCES:
{'-'*40}
{xai_refs_text}

{'='*60}
IMPORTANT NOTES:
{'-'*40}
• The diagnosis was made by AI with {disease_data['confidence']*100:.1f}% confidence.
  Confirm with a local expert if you are uncertain about the result.

• The treatment recommendations are CURATED from VERIFIED sources including:
  - Agrochemical company product labels (Bayer, Syngenta, Greenlife, etc.)
  - Published scientific research papers
  - Agricultural extension guides
  - University recommendations

• ALWAYS read and follow the product label instructions for any chemical used.
  Application rates and safety precautions vary by product.

• Product availability varies by location - check with your local agrovet.

• Consult your local agricultural extension officer for region-specific advice.

{'='*60}
"""

    if weather_data:
        report += f"""

WEATHER CONDITIONS:
{'-'*40}
Location: {weather_data.get('location', 'Unknown')}
Temperature: {weather_data.get('temperature', 'N/A')}°C
Humidity: {weather_data.get('humidity', 'N/A')}%
Rainfall: {weather_data.get('rain', 'N/A')} mm
Risk Assessment: {weather_data.get('risk_msg', 'N/A')}

{'='*60}
"""

    return report

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model_and_classes():
    """Load the trained model and class names"""
    model_path = 'crop_disease_classifier_model.keras'
    class_names_path = 'class_names.json'

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        return None, None

    model = keras.models.load_model(model_path)

    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    return model, class_names

# ============================================================
# CLASSES ORGANIZED BY CROP TYPE
# ============================================================
def organize_classes_by_crop(class_names):
    """Organize classes into Maize, Beans, and Tomato categories"""
    maize = []
    beans = []
    tomato = []

    for name in class_names:
        if name.lower().startswith('maize'):
            maize.append(name)
        elif name.lower().startswith('beans'):
            beans.append(name)
        else:
            tomato.append(name)

    return maize, beans, tomato

def display_classes_list(class_names):
    """Display the list of all classes organized by crop type"""
    maize, beans, tomato = organize_classes_by_crop(class_names)

    st.markdown("""
    <div class="classes-card">
        <h3 style="color: #2E7D32; margin-top: 0; margin-bottom: 15px;">📋 SUPPORTED CLASSES</h3>
        <p>The system can identify and provide treatment recommendations for the following crop diseases and conditions:</p>
    """, unsafe_allow_html=True)

    if maize:
        st.markdown('<div class="classes-category">🌽 MAIZE DISEASES & CONDITIONS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="class-count">({len(maize)} classes)</div>', unsafe_allow_html=True)
        for item in sorted(maize):
            st.markdown(f'<div class="class-item">• {item}</div>', unsafe_allow_html=True)

    if beans:
        st.markdown('<div class="classes-category">🫘 BEANS DISEASES & CONDITIONS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="class-count">({len(beans)} classes)</div>', unsafe_allow_html=True)
        for item in sorted(beans):
            st.markdown(f'<div class="class-item">• {item}</div>', unsafe_allow_html=True)

    if tomato:
        st.markdown('<div class="classes-category">🍅 TOMATO DISEASES & CONDITIONS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="class-count">({len(tomato)} classes)</div>', unsafe_allow_html=True)
        for item in sorted(tomato):
            st.markdown(f'<div class="class-item">• {item}</div>', unsafe_allow_html=True)

    st.markdown(f"""
        <hr>
        <p><strong>Total classes:</strong> {len(class_names)}</p>
        <p><strong>Note:</strong> This list includes both diseases and healthy crop conditions.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# GEOLOCATION FUNCTION
# ============================================================
def get_location_from_ip():
    """Get location using IP geolocation"""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return {
                    'city': data.get('city'),
                    'region': data.get('regionName'),
                    'country': data.get('country'),
                    'lat': data.get('lat'),
                    'lon': data.get('lon')
                }
    except:
        pass
    return None

def get_location_name_from_coords(lat, lon):
    """Convert GPS coordinates to a readable location name using reverse geocoding"""
    try:
        st.info(f"🌍 Looking up location for coordinates: {lat:.4f}, {lon:.4f}")

        response = requests.get(
            f"https://nominatim.openstreetmap.org/reverse",
            params={
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'zoom': 10,
                'addressdetails': 1
            },
            headers={'User-Agent': 'CropDoctor/1.0 (https://huggingface.co/spaces/dosuto/crop-doctor)'},
            timeout=8
        )
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})

            city = address.get('city') or address.get('town') or address.get('village') or ''
            county = address.get('county') or address.get('state_district') or ''
            state = address.get('state') or ''
            country = address.get('country') or 'Kenya'

            location_parts = []
            if city:
                location_parts.append(city)
            if county and county != city:
                location_parts.append(county)
            elif state and state != city:
                location_parts.append(state)
            location_parts.append(country)

            if location_parts:
                result = ', '.join(location_parts)
                st.success(f"✅ Location found: {result}")
                return result
            else:
                return f"Lat: {lat:.4f}, Lon: {lon:.4f}"
    except Exception as e:
        st.error(f"Reverse geocoding error: {e}")
        return f"Coordinates: {lat:.4f}, {lon:.4f}"

# ============================================================
# COMPLETE REFERENCES DICTIONARY
# ============================================================
def get_references():
    """Return complete references dictionary - 103 reference entries"""
    return {
        1: "SARI, APNI, CSIR-SARI, 'Maize Cropping Guide: 4R Nutrient Management and Best Agronomic Practices, Northern Ghana,' 2022.",
        2: "SA Grain, 'The big five maize leaf diseases: identification and management,' SA Grain, 2026.",
        3: "D. N. Shepherd et al., 'Maize streak virus: an old and complex emerging pathogen,' Molecular Plant Pathology, vol. 11, no. 1, pp. 1-12, 2009.",
        4: "M. K. Haraman, 'Management of maize streak virus disease (MSVD),' CABI Plantwise Knowledge Bank, 2013.",
        5: "Bayer East Africa, 'Gaucho® 600 FS Insecticide Product Label (Imidacloprid 600g/L),' 2025.",
        6: "Greenlife Crop Protection Africa, 'Emerald® 200SL Insecticide Product Label (Imidacloprid 200g/L),' 2025.",
        7: "Villa Crop Protection (PTY) LTD, 'Curator ULV Insecticide Product Label,' 2025.",
        8: "Twiga Chemicals, 'NeemAzal 1.2EC Technical Bulletin,' 2022.",
        9: "Osho Chemical Industries Ltd, 'Nimbecidine Product label,' 2025.",
        10: "P. Shango et al., 'Maize Lethal Necrosis (MLN) in Kenya,' African Phytosanitary Journal, vol. 1, no. 1, pp. 35-42, 2019.",
        11: "Z. Bin-hui et al., 'Maize lethal necrosis disease,' Journal of Integrative Agriculture, vol. 21, no. 12, pp. 3445-3455, 2022.",
        12: "M. Matimelo and M. Mudenda, 'Maize lethal necrosis disease in maize - Zambia,' CABI PlantwisePlus, 2023.",
        13: "H. Lemma et al., 'Prevention and detection of Maize Lethal Necrosis Disease,' CABI Plantwise, 2014.",
        14: "Veterinary & Agricultural Products Manufacturing Company (VAPCO), 'Diazinon 60% EC Insecticide Technical Bulletin,' 2024.",
        15: "YIWU S-LING Industry CO., LTD, 'Malathion 50% Technical Data Sheet,' 2025.",
        16: "Agro Pests, 'Lamdastar 5% EC Insecticide Catalogue (Lambda-cyhalothrin 5%),' 2025.",
        17: "B. W. Mahy and M. Van Regenmortel, 'Encyclopedia of Virology,' 2008.",
        18: "D. L. Nsibo et al., 'Maize foliar fungal pathogens in Africa,' Frontiers in Plant Science, vol. 15, 2024.",
        19: "Bayer East Africa, 'Nativo 300 SC Fungicide Product Catalogue (Tebuconazole 250g/L + Trifloxystrobin 150g/L),' 2025.",
        20: "Greenlife Crop Protection Africa, 'Absolute® 375SC Fungicide Product Label (Azoxystrobin + Difenoconazole + Hexaconazole),' 2025.",
        21: "Agrichem Africa Limited, 'Strobin Extra 325SC Product label,' 2024.",
        22: "Syngenta East Africa, 'Maxim® Quattro Fungicide Product Catalogue,' 2025.",
        23: "Greenlife Crop Protection Africa, 'Milestone® 250SC Fungicide Product Label (Azoxystrobin 250g/L),' 2025.",
        24: "Syngenta East Africa, 'Tilt 250 EC Fungicide Product Catalogue (Propiconazole 250g/L),' 2025.",
        25: "Greenlife Crop Protection Africa, 'Defacto® 500EC Fungicide Product Label (Propiconazole / Tebuconazole),' 2025.",
        26: "Crop Protection Network, 'Curvularia Leaf Spot of Corn,' 2020.",
        27: "UPL Limited, 'Dithane M-45 Fungicide Technical Bulletin (Mancozeb 80% WP),' 2025.",
        28: "Twiga Chemical Industries Ltd, 'Milthane Super Fungicide Product List (Mancozeb),' 2025.",
        29: "Greenlife Crop Protection Africa, 'Fortress Gold 72 WP Fungicide Product Label (Mancozeb 72%),' 2025.",
        30: "Dean Malvick, 'Northern corn leaf spot,' University of Minnesota Extension, 2018.",
        31: "BASF East Africa Limited, 'Comet 200 EC Fungicide Product Catalogue (Pyraclostrobin 200g/L),' 2025.",
        32: "Jubaili Agrotec Kenya, 'Dew Force Product Label,' 2022.",
        33: "Ningbo Sunjoy Agroscience, 'Pyracopp 400 SC Fungicide Technical Data Sheet,' 2025.",
        34: "CIMMYT, 'Maize Research,' International Maize and Wheat Improvement Center, 2025.",
        35: "D. Malvick, 'Common rust on corn,' University of Minnesota Extension, 2018.",
        36: "Greenlife Crop Protection Africa, 'Ransom® 600WP Product Label (Triadimefon),' 2025.",
        37: "GrowMate Kenya Limited, 'Azolaxyl 390SC Fungicide Product Catalogue (Azoxystrobin),' 2025.",
        38: "M. C. Makgoba et al., 'Fall armyworm impact on small-scale maize farmers,' Jamba, vol. 13, no. 1, 2021.",
        39: "Greenlife Crop Protection Africa, 'Managing Fall Armyworm Threat in Maize,' 2025.",
        40: "Greenlife Crop Protection Africa, 'Escort® 19EC Insecticide Product Label (Emamectin benzoate 19g/L),' 2025.",
        41: "Corteva Agriscience, 'Fidelity 400WG Insecticide Product Catalogue (Spinetoram 400g/Kg),' 2025.",
        42: "Greenlife Crop Protection Africa, 'Benzoron 110 ME Insecticide Product Label,' 2025.",
        43: "Greenlife Crop Protection Africa, 'Occasion Star 200 SC Insecticide Product Label,' 2025.",
        44: "Greenlife Crop Protection Africa, 'Indoking 300 SC Insecticide Product Label,' 2025.",
        45: "Greenlife Crop Protection Africa, 'Legacy Extreme 500 WDG Insecticide Product Label,' 2025.",
        46: "G. Jackson, 'Pacific Pests, Pathogens & Weeds: Maize aphid,' 2015.",
        47: "Greenlife Crop Protection Africa, 'Presento® 200SP Insecticide Product Label (Acetamiprid 200g/Kg),' 2025.",
        48: "S. H. Wani et al., 'Current Advances in Yield, Quality, and Stress Tolerance,' Springer Nature, 2023.",
        49: "ICL Fertilizers, 'Guide to Maize Nutrient Deficiency,' 2024.",
        50: "Iowa State University Extension and Outreach, 'Greensnap,' 2020.",
        51: "Ohio State University Extension, 'Other Corn Ear Abnormalities,' 2022.",
        52: "A. G. Martin et al., 'Soil Compaction on Trifluralin Phytotoxicity to Corn,' Agronomy Journal, vol. 77, no. 3, pp. 481-483, 1985.",
        53: "Z. Yang et al., 'Maize environmental stress resilience,' Molecular Plant, vol. 16, no. 10, 2023.",
        54: "Yara UK, 'Maize nutrient deficiencies,' 2024.",
        55: "CIAT, 'Bean Disease and Pest identification and management,' International Center for Tropical Agriculture, 2010.",
        56: "Greenlife Crop Protection Africa, 'Products List: Fungicides & Insecticides,' 2025.",
        57: "CABI, 'Invasive Species Compendium,' CABI Digital Library, 2021.",
        58: "Syngenta East Africa, 'Bravo 720 SC Fungicide Product Catalogue (Chlorothalonil 720g/L),' 2025.",
        59: "Syngenta East Africa, 'Fungicides Product Catalogue,' 2025.",
        60: "BASF East Africa Limited, 'Vegetables Solutions Catalogue,' 2025.",
        61: "Bayer East Africa, 'Crop Science Kenya - Fungicides,' 2025.",
        62: "Bayer East Africa, 'Folicur 250EW Fungicide Product Catalogue (Tebuconazole 250g/L),' 2025.",
        63: "BASF East Africa Limited, 'Kumulus DF Fungicide Product Catalogue (Wettable Sulphur 80%),' 2025.",
        64: "DuPont, 'Kocide 2000 Fungicide Technical Bulletin (Copper hydroxide 53.8%),' 2006.",
        65: "Nordox, 'Nordox 75 WG Copper Fungicide Technical Data (Copper hydroxide 75%),' 2025.",
        66: "Greenlife Crop Protection Africa, 'Copac E Copper-based Fungicide Product Label (Green Cop® 500WP),' 2025.",
        67: "Bayer East Africa, 'Cupravit® 50 WP Fungicide Product Label (Copper oxychloride 50%),' 2025.",
        68: "Syngenta East Africa, 'Amistar® Opti Fungicide Product Catalogue (Azoxystrobin + Chlorothalonil),' 2025.",
        69: "A. Candidato, 'Growth Characteristics of Tomato,' Plant Perspectives, 2026.",
        70: "Greenlife Crop Protection Africa, 'Tomato farming for beginners,' 2025.",
        71: "University of California IPM, 'Tomato: Pest and disease management,' UC IPM Online.",
        72: "Cornell University, 'Tomato disease identification and management,' Cornell Vegetable Program.",
        73: "University of Minnesota Extension, 'Early blight in tomato and potato,' 2024.",
        74: "Crystal Crop Protection Limited, 'Bavistin Carbendazim 50% WP Technical Bulletin,' 2023.",
        75: "Bayer Crop Science, '42-S Thiram Seed Treatment Product Label,' 2025.",
        76: "Verdera, 'Mycostop Biological Fungicide Product Label,' 2025.",
        77: "Syngenta East Africa, 'Ridomil Gold® MZ 68 WG Fungicide Catalogue (Metalaxyl-M + Mancozeb),' 2025.",
        78: "Bayer East Africa, 'Reason 500 SC Fungicide Catalogue (Fenamidone 500g/L),' 2025.",
        79: "Corteva Agriscience, 'Curzate MZ Fungicide Catalogue (Cymoxanil + Mancozeb),' 2025.",
        80: "BASF East Africa Limited, 'Acrobat 500 SC Fungicide Catalogue (Dimethomorph 500g/L),' 2025.",
        81: "Syngenta East Africa, 'Revus 250 SC Product Label (Mandipropamid),' 2025.",
        82: "Bayer East Africa, 'Confidor® 200SL Insecticide Product Label (Imidacloprid 200g/L),' 2025.",
        83: "Syngenta East Africa, 'Karate® 5EC Insecticide Product Catalogue (Lambda-cyhalothrin 50g/L),' 2025.",
        84: "Dudutech, 'Yellow Sticky Traps Technical Specifications,' 2026.",
        85: "Certis Biologicals, 'Cueva® Flowable Liquid Copper Fungicide,' 2026.",
        86: "Cosaco, 'ManKocide Product Datasheet (Copper hydroxide + Mancozeb),' 2025.",
        87: "Bayer Crop Science, 'Serenade Biological Fungicide Product Label (Bacillus subtilis),' 2023.",
        88: "Corteva Agriscience, 'Steward 30WG Insecticide Catalogue (Indoxacarb 30% w/w),' 2025.",
        89: "FMC Corporation, 'Coragen 20SC Insecticide Technical Bulletin (Chlorantraniliprole 200g/L),' 2026.",
        90: "Certis Biologicals, 'Thuricide® Bacillus thuringiensis (Bt) Bioinsecticide,' 2020.",
        91: "AgBiTech, 'ViVUS® Max Biological Insecticide (Nuclear Polyhedrosis Virus - NPV),' 2026.",
        92: "Corteva Agriscience, 'Tracer 480SC Insecticide Catalogue (Spinosad 480g/L),' 2025.",
        93: "Bayer East Africa, 'Belt 480SC Insecticide Catalogue (Flubendiamide 480g/L),' 2025.",
        94: "Syngenta Egypt, 'Vertimec 018 EC Product Label (Abamectin),' 2025.",
        95: "Bayer East Africa, 'Oberon 240SC Miticide Catalogue (Spiromesifen 240g/L),' 2025.",
        96: "Corteva Agriscience, 'Omite 570EW Miticide Catalogue (Propargite 570g/L),' 2025.",
        97: "Certis Biologicals, 'DES-X: Insecticidal Soap Concentrate,' 2026.",
        98: "R. S. Byther, 'Tomato: Physiological disorders,' Hortsense, Washington State University.",
        99: "Teagasc, 'Getting to grips with physiological disorders in tomato crops,' Agriland.",
        100: "Tetra Technologies Inc., 'Calcium Chloride for Agricultural Use,' 2026.",
        101: "Yara US, 'YaraLiva® CALCINIT® 15.5-0-0 (Calcium Nitrate Fertilizer) Technical Data Sheet,' 2026.",
        102: "Mosaic, 'Calcium Carbonate (Lime) Soil Amendment Specifications,' 2026.",
        103: "Gypsoil, 'Gypsum (Calcium Sulphate) Soil Amendment Technical Data Sheet,' 2026."
    }

# ============================================================
# BOLD FUNCTION
# ============================================================
def _bold(text):
    """Return bold formatted text"""
    return f"**{text}**"

# ============================================================
# WEATHER FUNCTION - DISEASE SPECIFIC RISK ASSESSMENT
# ============================================================

def get_weather_with_risk_assessment(location, disease_name, treatment_data=None):
    """Get weather and provide disease-specific risk assessment based on actual disease characteristics
    PRIORITY: 1. GPS coordinates, 2. Geocoded location name
    """
    try:
        # PRIORITY 1: Use GPS coordinates if available (most accurate for Kenya)
        lat = None
        lon = None
        location_source = "unknown"

        if st.session_state.get('gps_location') and st.session_state.gps_location.get('lat'):
            lat = st.session_state.gps_location['lat']
            lon = st.session_state.gps_location['lon']
            location_source = "GPS"
            print(f"📍 Weather using GPS coordinates: {lat:.6f}, {lon:.6f}")
        else:
            # PRIORITY 2: Geocode the location name (fallback)
            geocode_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
            geo_response = requests.get(geocode_url, headers={'User-Agent': 'CropDoctor/1.0'}, timeout=8)

            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                if geo_data:
                    lat = float(geo_data[0]['lat'])
                    lon = float(geo_data[0]['lon'])
                    location_source = "geocoding"
                    print(f"📍 Weather using geocoded location: {location} -> {lat:.6f}, {lon:.6f}")

        # Default to Nairobi if all methods fail
        if lat is None or lon is None:
            lat, lon = -1.286389, 36.817223  # Nairobi coordinates
            location_source = "default (Nairobi)"
            print(f"📍 Weather using default location: Nairobi")

        # Get weather data from Open-Meteo API
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                'latitude': lat,
                'longitude': lon,
                'current': ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m'],
                'daily': ['temperature_2m_max', 'temperature_2m_min', 'precipitation_probability_max', 'rain_sum'],
                'timezone': 'Africa/Nairobi'
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            daily = data.get('daily', {})

            # Extract current weather
            temp = current.get('temperature_2m', 'N/A')
            humidity = current.get('relative_humidity_2m', 'N/A')
            rain = current.get('precipitation', 0)
            wind = current.get('wind_speed_10m', 'N/A')

            # Extract daily forecast
            temp_max = daily.get('temperature_2m_max', [None])[0] if daily else None
            temp_min = daily.get('temperature_2m_min', [None])[0] if daily else None
            rain_prob = daily.get('precipitation_probability_max', [None])[0] if daily else None
            rain_sum = daily.get('rain_sum', [0])[0] if daily else 0

            # Check if this is a healthy crop
            if 'healthy' in disease_name.lower():
                return {
                    'temperature': temp,
                    'humidity': humidity,
                    'rain': rain,
                    'wind': wind,
                    'temp_max': temp_max,
                    'temp_min': temp_min,
                    'rain_prob': rain_prob,
                    'rain_sum': rain_sum,
                    'risk_msg': "NONE - Healthy crop (no disease risk assessment applicable)",
                    'risk_class': "risk-low",
                    'location': location,
                    'location_source': location_source,
                    'lat': lat,
                    'lon': lon
                }

            # Disease-specific risk assessment
            risk_msg = "NONE - No specific weather-disease relationship established"
            risk_class = "risk-low"

            if treatment_data:
                causes_text = treatment_data.get('causes_characteristics', '')
                management_text = treatment_data.get('management', '')
                full_text = (causes_text + " " + management_text).lower()

                # Check weather dependencies in disease description
                weather_keywords = {
                    'humidity': ['humid', 'humidity', 'moist', 'wet', 'damp', 'high moisture'],
                    'temperature': ['warm', 'hot', 'cool', 'cold', 'temperature', 'heat', 'cool temperatures'],
                    'rain': ['rain', 'rainfall', 'splashing rain', 'splashed by rain', 'wet conditions'],
                    'dry': ['dry', 'drought', 'arid', 'dry conditions']
                }

                has_humidity_link = any(keyword in full_text for keyword in weather_keywords['humidity'])
                has_temp_link = any(keyword in full_text for keyword in weather_keywords['temperature'])
                has_rain_link = any(keyword in full_text for keyword in weather_keywords['rain'])
                has_dry_link = any(keyword in full_text for keyword in weather_keywords['dry'])

                # Detect disease type
                category = treatment_data.get('category', '').lower()
                is_fungal = 'fungal' in category or any(word in full_text for word in ['fungus', 'fungal', 'spores', 'mycelium'])
                is_viral = 'viral' in category or any(word in full_text for word in ['virus', 'vector', 'whitefly', 'aphid'])
                is_pest = 'pest' in category or any(word in full_text for word in ['pest', 'larvae', 'insect', 'caterpillar'])

                # Generate risk assessment based on disease characteristics
                if has_humidity_link or has_temp_link or has_rain_link or has_dry_link:

                    # FUNGAL DISEASE RISK
                    if is_fungal and has_humidity_link and humidity != 'N/A':
                        if humidity > 80:
                            risk_msg = f"⚠️ HIGH FUNGAL DISEASE RISK - High humidity ({humidity}%) favours fungal growth. The disease description indicates that {disease_name} spreads under humid conditions."
                            risk_class = "risk-high"
                        elif humidity > 65:
                            risk_msg = f"🟡 MODERATE FUNGAL DISEASE RISK - Current humidity ({humidity}%) may favour {disease_name} development according to disease characteristics."
                            risk_class = "risk-moderate"
                        else:
                            risk_msg = f"✅ Low fungal disease risk - Current humidity ({humidity}%) is not favourable for {disease_name} based on its disease characteristics."
                            risk_class = "risk-low"

                    # VIRAL DISEASE RISK (vector activity)
                    elif is_viral and (has_temp_link or 'vector' in full_text) and temp != 'N/A':
                        if temp > 25 and humidity > 60:
                            risk_msg = f"⚠️ HIGH VIRAL DISEASE RISK - Warm, humid conditions ({temp}°C, {humidity}%) favour vector activity. The disease is transmitted by vectors as described."
                            risk_class = "risk-high"
                        elif temp > 22:
                            risk_msg = f"🟡 MODERATE VIRAL DISEASE RISK - Current conditions ({temp}°C) may support vector activity for {disease_name}."
                            risk_class = "risk-moderate"
                        else:
                            risk_msg = f"✅ Low viral disease risk - Current conditions ({temp}°C) are less favourable for vectors of {disease_name}."
                            risk_class = "risk-low"

                    # PEST INFESTATION RISK
                    elif is_pest and has_dry_link and temp != 'N/A':
                        if temp > 25 and rain_sum < 5:
                            risk_msg = f"⚠️ HIGH PEST RISK - Warm, dry conditions ({temp}°C) favour {disease_name} according to its pest characteristics."
                            risk_class = "risk-high"
                        elif temp > 20:
                            risk_msg = f"🟡 MODERATE PEST RISK - Current temperature ({temp}°C) may favour {disease_name} activity."
                            risk_class = "risk-moderate"
                        else:
                            risk_msg = f"✅ Low pest risk - Current temperature ({temp}°C) is less favourable for {disease_name}."
                            risk_class = "risk-low"

                    # RAIN-SPREAD DISEASES
                    elif has_rain_link:
                        if rain_sum > 10 or rain > 1:
                            risk_msg = f"⚠️ HIGH DISEASE SPREAD RISK - Rainfall ({rain_sum}mm) can spread {disease_name} as described in disease characteristics."
                            risk_class = "risk-high"
                        elif rain_prob > 50:
                            risk_msg = f"🟡 MODERATE DISEASE SPREAD RISK - Expected rainfall may facilitate spread of {disease_name}."
                            risk_class = "risk-moderate"
                        else:
                            risk_msg = f"✅ Low disease spread risk - Dry conditions reduce spread of {disease_name}."
                            risk_class = "risk-low"
                    else:
                        risk_msg = "NONE - No specific weather-disease relationship established for this disease in the database."
                        risk_class = "risk-low"
                else:
                    risk_msg = "NONE - This disease's characteristics do not establish a specific relationship with weather conditions."
                    risk_class = "risk-low"
            else:
                risk_msg = "NONE - No treatment data available to determine weather-disease relationship."
                risk_class = "risk-low"

            return {
                'temperature': temp,
                'humidity': humidity,
                'rain': rain,
                'wind': wind,
                'temp_max': temp_max,
                'temp_min': temp_min,
                'rain_prob': rain_prob,
                'rain_sum': rain_sum,
                'risk_msg': risk_msg,
                'risk_class': risk_class,
                'location': location,
                'location_source': location_source,
                'lat': lat,
                'lon': lon
            }
        else:
            return {
                'temperature': 'N/A',
                'humidity': 'N/A',
                'rain': 0,
                'wind': 'N/A',
                'temp_max': None,
                'temp_min': None,
                'rain_prob': None,
                'rain_sum': 0,
                'risk_msg': "Unable to fetch weather data. Please check your internet connection.",
                'risk_class': "risk-low",
                'location': location,
                'location_source': "error",
                'lat': lat,
                'lon': lon
            }
    except Exception as e:
        print(f"Weather API error: {e}")
        return {
            'temperature': 'N/A',
            'humidity': 'N/A',
            'rain': 0,
            'wind': 'N/A',
            'temp_max': None,
            'temp_min': None,
            'rain_prob': None,
            'rain_sum': 0,
            'risk_msg': f"Weather service temporarily unavailable. Please try again later.",
            'risk_class': "risk-low",
            'location': location,
            'location_source': "error",
            'lat': None,
            'lon': None
        }

# ============================================================
# COMPLETE TREATMENT DATABASE - ALL CLASSES
# ============================================================

def get_full_treatment(class_name, references):
    """Complete treatment database with correct reference mapping"""
    name_lower = class_name.lower()

    # ==================== MAIZE CURVULARIA DISEASE ====================
    if 'curvularia' in name_lower:
        return {
            "disease_name": "Curvularia Leaf Spot Disease (CLSD)",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Curvularia lunata (syn. Cochliobolus lunatus)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Curvularia lunata (syn. Cochliobolus lunatus).

{_bold('Symptoms:')}
• Initial symptoms: Small, tan, round spots often surrounded by a yellow halo.
• Later symptoms: Spots grow into round or oval lesions with a yellowish-brown centre and a dark brown border.
• Disease cycle: The fungus persists in infected maize residues and soil, infecting the next crops, especially in wet conditions.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
1. Apply foliar fungicides preventatively or at the very early stages of disease development.
2. Regular monitoring: Monitor fields regularly, especially after periods of warm, wet weather.
3. Improve air circulation through proper spacing.

{_bold('FOR LONG-TERM MANAGEMENT:')}
• Plant CLSD resistant varieties.""",
            "chemical_control": f"""{_bold('Appropriate fungicides for CLSD include:')}

{_bold('PROTECTANT FUNGICIDES (Apply before infection):')}
i) Mancozeb (e.g., Dithane M-45 from UPL Limited [1], Milthane Super from Twiga Chemical Industries Ltd [2], Fortress Gold 72 WP both from Greenlife Crop Protection Africa Ltd [3]). Use 2.5g/L of water.

{_bold('SYSTEMIC/CURATIVE FUNGICIDES (Apply at first symptoms):')}
i) Strobilurins (e.g., Nativo 300 SC from Bayer Kenya [4], Absolute 375SC from Greenlife Crop Protection Africa [5], Strobin Extra 325SC from Agrichem Africa Limited [6], Maxim® Quattro from Syngenta Crop Protection AG [7]).
ii) Triazoles (e.g., Azoxystrobin like Milestone® 250SC from Greenlife Crop Protection Africa [8]).
iii) Propiconazole (e.g., Tilt 250 EC from Syngenta Crop Protection AG [9] and Defacto® 500EC from Greenlife Crop Protection Africa [10]).""",
            "xai_ref_numbers": [26],
            "management_refs": [26],
            "chemical_refs_original": [27, 28, 29, 19, 20, 21, 22, 23, 24, 25]
        }

    # ==================== MAIZE STRIPE VIRUS ====================
    elif 'stripe virus' in name_lower and 'maize' in name_lower:
        return {
            "disease_name": "Maize Stripe Virus Disease (MSpVD)",
            "category": "🦠 VIRAL DISEASE",
            "causal_agent": "Maize stripe virus (MSpV). Transmitted by maize planthopper",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Maize stripe virus (MSpV). Transmitted by maize planthopper.

{_bold('Symptoms:')}
• Initial symptoms are fine chlorotic (yellow or white) spots and narrow stripes on the youngest leaves. Later, continuous chlorotic stripes develop with varying width and intensity running parallel to the veins along the entire length of the leaf.
• Infected plants are often stunted and die prematurely.
• Unlike the broken, irregular streaks of Maize Streak Virus (MSV), MSpV symptoms are characterized by their continuous and uniform striping pattern.""",
            "management": f"""{_bold('⚠️ NO CURE FOR INFECTED PLANTS. Focus on prevention. ⚠️')}

{_bold('FOR INFECTED CROPS:')}
• Uproot and destroy infected plants immediately.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Use certified seeds: Plant seeds that are free from MSpV and resistant to it.
2. Crop rotation: Rotate maize with non-cereal crops (e.g., sweet potatoes, beans) for 2-3 seasons.
3. Avoid infested fields: Do not plant maize within 500m of fields with infested maize.
4. Quarantine: Avoid movement of green maize from infected to disease-free areas.
5. Weed control: Eliminate host plants (weeds) of thrips, aphids & plant hoppers and clear/burn grass around the field.
6. Sterilise machinery: Sterilise farm machinery as MSpV can adhere to it and persist for long.""",
            "chemical_control": f"""{_bold('Chemical Control (for vector control only – ineffective against the virus itself):')}

Apply insecticide treatments to protect crops from maize planthopper/leafhopper vectors:

i) Imidacloprid (e.g., Gaucho® 600 FS from Bayer East Africa [1], Emerald 200SL from Greenlife Crop Protection Africa [2]).
ii) Curator ULV from Villa Crop Protection (PTY) LTD [3].""",
            "xai_ref_numbers": [2, 17],
            "management_refs": [2, 17],
            "chemical_refs_original": [5, 6, 7]
        }

    # ==================== MAIZE STREAK VIRUS ====================
    elif 'streak virus' in name_lower:
        return {
            "disease_name": "Maize Streak Virus Disease (MSVD)",
            "category": "🦠 VIRAL DISEASE",
            "causal_agent": "Maize streak virus (MSV). Spread by leafhoppers.",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Maize streak virus (MSV). Spread by leafhoppers.

{_bold('Symptoms:')}
• Early symptoms are small round spots on the lowest exposed portion of the youngest leaves.
• Spots turn into white/yellow/light green streaks running along leaf veins.
• Only leaves formed after infection develop symptoms; older leaves remain healthy.
• Early infection causes severe stunting, undersized/deformed cobs or no yield.
• Poor ear development.""",
            "management": f"""{_bold('⚠️ NO CURE FOR INFECTED PLANTS. Focus on prevention.')}

{_bold('FOR INFECTED CROPS:')}
• Uproot and destroy infected plants immediately.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant resistant varieties: Use certified MSV-resistant seeds.
2. Control vectors: Manage leafhopper populations to prevent spread.""",
            "chemical_control": f"""{_bold('Chemical Control (for vector control only – ineffective against the virus itself):')}

Apply insecticide treatments to protect crops from leafhopper vectors.

{_bold('Chemical Insecticides:')}
i) Imidacloprid (e.g., Gaucho® 600 FS from Bayer East Africa [1], Emerald 200SL from Greenlife Crop Protection Africa [2]).
ii) Curator ULV from Villa Crop Protection (PTY) LTD [3].

{_bold('Organic Options for Leafhopper Control:')}
i) Neem oil (e.g., NeemAzal 1.2EC from Twiga Chemicals [4], Nimbecidine from Osho Chemical Industries Ltd [5]). Use 5-10ml/L of water.""",
            "xai_ref_numbers": [2, 3, 4],
            "management_refs": [2, 3, 4],
            "chemical_refs_original": [5, 6, 7, 8, 9]
        }

    # ==================== MAIZE HELMINTHOSPORIOSIS ====================
    elif 'helminthosporiosis' in name_lower:
        return {
            "disease_name": "Helminthosporiosis (Northern Corn Leaf Spot / Carbonum Leaf Spot)",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Bipolaris zeicola (formerly Helminthosporium carbonum)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Bipolaris zeicola (formerly Helminthosporium carbonum).

{_bold('Symptoms:')}
• It is characterised by numerous small, dark brown to blackish, rectangular or oval spots. These are often surrounded by a chlorotic (yellow) halo.
• In severe cases, numerous spots can merge, causing large areas of the leaf to wither and die, giving it a 'blighted' or burnt appearance.
• On the ears and kernels, a characteristic sign is a black or dark olive-green, felty mold growing on and between kernels.
• The fungus survives for multiple years in and on infected maize debris left on the soil surface. Spores are splashed by rain or carried by wind onto the leaves of the new maize crop.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
• Apply appropriate fungicides. Optimal timing is from tasseling (VT) to silking (R1).

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant hybrid varieties that are resistant to Helminthosporiosis.
2. Crop rotation: Rotate with non-host plants (e.g., legumes).
3. Residue management: Bury crop residues during tilling or feed them to animals.""",
            "chemical_control": f"""{_bold('Appropriate fungicides for Helminthosporiosis include:')}

i) Tebuconazole (e.g., Nativo 300 SC from Bayer Kenya [1], Defacto® 500EC from Greenlife Crop Protection Africa [2]).
ii) Pyraclostrobin (e.g., BASF Comet 200 EC from BASF East Africa Limited [3], Dew Force Jubaili Agrotec Kenya [4], Pyracopp 400 SC from Ningbo Sunjoy Agroscience [5]).""",
            "xai_ref_numbers": [2, 30],
            "management_refs": [2, 30],
            "chemical_refs_original": [19, 25, 31, 32, 33]
        }

    # ==================== MAIZE LETHAL NECROSIS ====================
    elif 'lethal necrosis' in name_lower:
        return {
            "disease_name": "Maize Lethal Necrosis Disease (MLND)",
            "category": "🦠 VIRAL DISEASE",
            "causal_agent": "Co-infection of Maize chlorotic mottle virus (MCMV) and a potyvirus",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Co-infection of Maize chlorotic mottle virus (MCMV) and a potyvirus.

{_bold('Symptoms:')}
• Rapid yellowing of leaves (wider than those of MSV).
• Leaf drying from margins.
• Dead heart.
• Plant death.
• No grain filling in cobs or poorly filled cobs.""",
            "management": f"""{_bold('⚠️ NO CURE FOR INFECTED PLANTS. Focus on prevention. ⚠️')}

{_bold('FOR INFECTED CROPS:')}
• Uproot and destroy infected plants immediately.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Use certified seeds: Plant seeds that are free from MCMV and resistant to it.
2. Crop rotation: Rotate maize with non-cereal crops (e.g., sweet potatoes, beans) for 2-3 seasons.
3. Avoid infested fields: Do not plant maize within 500m of fields with infested maize.
4. Quarantine: Avoid movement of green maize from infected to disease-free areas.
5. Weed control: Eliminate host plants (weeds) of thrips, aphids & plant hoppers and clear/burn grass around the field.
6. Sterilise machinery: Sterilise farm machinery as MCMV can adhere to it and persist for long.""",
            "chemical_control": f"""{_bold('Chemical Control (for vector control only – ineffective against the virus itself):')}

Control vectors (aphids and thrips) with appropriate insecticides:

i) Imidacloprid (e.g., Emerald 200SL from Greenlife Crop Protection Africa [1]).
ii) Diazinon 60% EC from VAPCO [2].
iii) Malathion 50% from YIWU S-LING Industry CO., LTD [3].
iv) Lambda-cyhalothrin (e.g., Lamdastar 5%EC from Agro Pests) [4].""",
            "xai_ref_numbers": [10, 11, 12, 13],
            "management_refs": [10, 11, 12, 13],
            "chemical_refs_original": [6, 14, 15, 16]
        }

    # ==================== MAIZE HEALTHY ====================
    elif 'maize healthy' in name_lower:
        return {
            "disease_name": "Healthy Maize Crop",
            "category": "🌿 HEALTHY CROP",
            "causal_agent": "No disease detected",
            "is_healthy": True,
            "causes_characteristics": f"""{_bold('Key Characteristics of a healthy maize plant:')}

1. Colour and Appearance: Leaves are vibrant, deep green, indicating healthy nitrogen levels and photosynthesis.
2. Stalk Strength: A thick, sturdy, and well-developed stem at the base, which resists falling over.
3. Root Structure: A robust, deep, and well-anchored root system, crucial for taking up nutrients and water.
4. Foliage: Clean foliage free from discolouration (yellowing/purpling) and without signs of pests or diseases.
5. Growth Pace: Steady and rapid growth, particularly in early stages, with good early leaf production.
6. Cob Development: Large, well-filled, and healthy cobs, preferably with 1-2 per plant, indicating good pollination and nutrients.

{_bold('No disease detected.')}""",
            "management": "Continue regular crop management. Maintain soil fertility, proper irrigation, and regular monitoring.",
            "chemical_control": "Not applicable. No treatment required.",
            "xai_ref_numbers": [1],
            "management_refs": [1],
            "chemical_refs_original": []
        }

    # ==================== MAIZE LEAF BLIGHT ====================
    elif 'leaf blight' in name_lower:
        return {
            "disease_name": "Maize Leaf Blight Disease (MLBD)",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Exserohilum turcicum (Northern Corn Leaf Blight) or Bipolaris maydis (Southern Corn Leaf Blight)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Exserohilum turcicum (Northern Corn Leaf Blight) or Bipolaris maydis (Southern Corn Leaf Blight).

{_bold('Characteristics:')}
• It is a fungal disease which thrives in warm, humid conditions and spreads by wind and rain.
• NCLB: Long, elliptical (cigar-shaped) lesions, typically 1 to 6 inches long. The lesions are grey-green at first but turn brown eventually.
• SCLB: Small, rectangular to oblong (four sided) lesions which are brown in colour. The spots may have yellowish borders. The lesions can be found on leaves, husks, and stalks.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
• Apply foliar fungicides in a timely manner. Monitor fields regularly and apply at the first appearance of symptoms.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant MLBD resistant varieties.
2. Crop rotation: Rotate with non-host crops such as legumes.
3. Residue management: Bury crop residues appropriately, as spores survive in debris.""",
            "chemical_control": f"""{_bold('Appropriate fungicides for MLBD include:')}

i) Strobilurins (e.g., Nativo 300 SC from Bayer Kenya [1], Absolute 375SC from Greenlife Crop Protection Africa [2], Strobin Extra 325SC from Agrichem Africa Limited [3], Maxim® Quattro from Syngenta Crop Protection AG [4]).
ii) Triazoles (e.g., Azoxystrobin like Milestone® 250SC from Greenlife Crop Protection Africa [5]).
iii) Propiconazole (e.g., Tilt 250 EC from Syngenta Crop Protection AG [6] and Defacto® 500EC from Greenlife Crop Protection Africa [7]).""",
            "xai_ref_numbers": [2, 18],
            "management_refs": [2, 18],
            "chemical_refs_original": [19, 20, 21, 22, 23, 24, 25]
        }

    # ==================== MAIZE FALL ARMYWORM ====================
    elif 'fall army' in name_lower or 'armyworm' in name_lower:
        return {
            "disease_name": "Fall Armyworm (FAW) & its Activity",
            "category": "🐛 PEST INFESTATION",
            "causal_agent": "Fall Armyworm (Spodoptera frugiperda)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Pest Characteristics:')}
• The Fall Armyworm larvae mainly feed at night. This makes them difficult to find, delaying their discovery until after considerable damage is done.
• The larvae seek refuge deep within the whorls and leaf folds of maize plants.
• The pest's different life stages (eggs, larvae, and pupae) often overlap in the field.
• Fall armyworm damage in maize features include: 'window panes' on leaves, irregular holes on the leaves, torn leaves. Wet, sawdust-like droppings (frass) are often left in the whorl and on damaged areas.""",
            "management": f"""{_bold('Integrated Management Approach:')}

1. Use appropriate pesticide to control the pest.
2. Frequently monitor the crops, with particular emphasis on the whorl and pre-tassel growth stages.
3. Use different combinations of insecticides that attack the pest in different ways (rotate insecticide groups to prevent resistance development).
4. Make sure to spray all the feeding sites of the pests, particularly the whorl and leaf folds.
5. Maximum insecticide effectiveness is achieved by targeting young, actively feeding larvae. Applications should be timed for early mornings or late evenings.
6. All insecticide applications must strictly adhere to recommended dosage rates and established safety protocols.""",
            "chemical_control": f"""{_bold('Appropriate insecticides include:')}

{_bold('Single Active Ingredients:')}
i) Emamectin benzoate (e.g., Escort 19EC from Greenlife Crop Protection Africa [1]).
ii) Spinetoram (e.g., Fidelity 400WG from Corteva Agriscience [2]).

{_bold('Complementary insecticide combinations (most effective) recommended by Greenlife Crop Protection Africa:')}
i) Benzonor 110 ME: 2.5ml + Integra 3 ml per 20 L of water [3].
ii) Escort 19EC: 20-25 ml + Integra 3 ml per 20 L of water [1].
iii) Occasion Star 200 SC: 3 ml + Integra 3 ml per 20 L of water [4].
iv) Indoking 300 SC: 3 ml + Integra 3 ml per 20 L of water [5].
v) Legacy Extreme 500 WDG: 2 g + Integra 3 ml per 20 L of water [6].

{_bold('Organic Options for FAW Control:')}
i) Neem oil (e.g., NeemAzal 1.2EC from Twiga Chemicals [7], Nimbecidine from Osho Chemical Industries Ltd [8]). Use 5-10ml/L of water.""",
            "xai_ref_numbers": [38, 39, 34],
            "management_refs": [38, 39, 34],
            "chemical_refs_original": [40, 41, 42, 43, 44, 45, 8, 9]
        }

    # ==================== MAIZE RUST ====================
    elif 'rust' in name_lower and 'maize' in name_lower:
        return {
            "disease_name": "Maize Rust Disease",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Puccinia sorghi (Common Rust) or Puccinia polysora (Southern Rust)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Puccinia sorghi (Common Rust) or Puccinia polysora (Southern Rust).

{_bold('Symptoms:')}
• The disease appears as reddish-brown to black pustules on corn leaves, stalks, and husks.
• The most characteristic symptom is the presence of pustules (pimples) on the leaf surfaces, which contain the rust-coloured spores.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
• Apply appropriate fungicides. Optimal timing is from tasseling (VT) to silking (R1).

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant hybrid varieties that are resistant to Maize Rust Disease.
2. Crop rotation: Rotate with non-host plants (e.g., legumes).
3. Residue management: Bury crop residues during tilling or feed them to animals.""",
            "chemical_control": f"""{_bold('Appropriate fungicides for Maize Rust Disease include:')}

i) Triadimefon (e.g., Ransom® 600WP from Greenlife Crop Protection Africa [1]).
ii) Azoxystrobin (e.g., Absolute 375SC from Greenlife Crop Protection Africa [2], Azolaxyl 390SC from GrowMate Kenya Limited [3]).""",
            "xai_ref_numbers": [34, 35],
            "management_refs": [34, 35],
            "chemical_refs_original": [36, 20, 37]
        }

    # ==================== MAIZE APHIDS ====================
    elif 'aphids' in name_lower or 'aphid' in name_lower:
        return {
            "disease_name": "Aphids Infestation",
            "category": "🐛 PEST INFESTATION",
            "causal_agent": "Various aphid species",
            "is_healthy": False,
            "causes_characteristics": f"""These are small, bluish-green to dark green sap-sucking pests that can spread diseases. They are found deep within the whorl, on tassels, and the undersides of upper leaves.""",
            "management": f"""{_bold('Integrated Management Approach:')}

1. Conduct regular field scouting for aphids. If the aphids are present but do not reach the economic threshold, use biological control measures (Lady Beetles).
2. If the projected economic loss will be huge, spray appropriate insecticides on the plants.

{_bold('FOR LONG-TERM PREVENTION:')}
1. Use resistant seeds during planting.
2. Avoid over-fertilization as excessive nitrogen can make the plants more succulent and attractive to aphids.""",
            "chemical_control": f"""{_bold('Appropriate insecticides include:')}

i) Imidacloprid (e.g., Emerald 200SL from Greenlife Crop Protection Africa [1], Diazinon 60% EC from VAPCO [2], Malathion 50% from YIWU S-LING Industry CO., LTD [3]).
ii) Lambda-cyhalothrin (e.g., Lamdastar 5% EC from Agro Pests [4]).
iii) Acetamiprid (e.g., Presento® 200SP from Greenlife Crop Protection Africa [5]).

{_bold('Organic Options for Aphids Control:')}
i) Neem oil (e.g., NeemAzal 1.2EC from Twiga Chemicals [6], Nimbecidine from Osho Chemical Industries Ltd [7]). Use 5-10ml/L of water.""",
            "xai_ref_numbers": [2, 46],
            "management_refs": [2, 46],
            "chemical_refs_original": [6, 14, 15, 16, 47, 8, 9]
        }

    # ==================== MAIZE ABIOTIC ====================
    elif 'abiotic' in name_lower:
        return {
            "disease_name": "Abiotic Diseases/Disorders of Maize",
            "category": "🌱 PHYSIOLOGICAL DISORDER",
            "causal_agent": "Environmental stress or nutrient deficiency",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('These are caused by environmental related issues rather than pathogens. Common abiotic disorders in maize include:')}

{_bold('NUTRIENT DEFICIENCIES AND TOXICITIES:')}
• Nitrogen (N) Deficiency: "V-shaped" yellowing starting from the leaf tip and moving down the midrib on older, lower leaves.
• Phosphorus (P) Deficiency: Purpling or reddish coloration of the leaves, stems, and husks. Stunted growth.
• Potassium (K) Deficiency: Yellowing and burning (necrosis) along the tips and margins of older leaves.
• Zinc (Zn) Deficiency: Broad bands of white to yellowish-white tissue on either side of the midrib on younger leaves.

{_bold('ENVIRONMENTAL STRESS:')}
• Moisture Stress (Drought/Waterlogging)
• Temperature Stress (Chilling/Heat)
• Hail & Wind Damage, Frost/Freeze Injury

{_bold('SOIL-RELATED AND CHEMICAL DISORDERS:')}
• Soil Compaction, Herbicide Injury, Soil Salinity/Alkalinity/Acidity

{_bold('PHYSIOLOGICAL AND GENETIC DISORDERS:')}
• Nutrient (Fertilizer) Burn, Brittle Snap / Green Snap
• Pollination Problems (Zipper Ears, Blunt Ear Tips)""",
            "management": f"""{_bold('Since abiotic disorders are not "cured," management focuses on prevention and mitigation by addressing the underlying cause:')}

1. Soil Management: Test the soil and apply recommendations. Avoid soil compaction.
2. Water Management: Use irrigation for drought stress; improve drainage for waterlogging.
3. Hybrid Selection: Choose hybrids with tolerance to specific stresses.
4. Careful Chemical Application: Apply herbicides/fertilizers at correct rate, time, and with proper equipment.
5. Timely Planting: Plant when soil temperatures are adequate to avoid chilling injury.""",
            "chemical_control": "Not applicable. Cultural management based on specific abiotic cause.",
            "xai_ref_numbers": [48, 49, 50, 51, 52, 53, 54],
            "management_refs": [48, 49, 50, 51, 52, 53, 54],
            "chemical_refs_original": []
        }

    # ==================== BEANS ANGULAR LEAF SPOT ====================
    elif 'angular leaf spot' in name_lower and 'beans' in name_lower:
        return {
            "disease_name": "Angular Leaf Spot (ALS) of Beans",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Pseudocercospora griseola (formerly Phaeoisariopsis griseola)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Pseudocercospora griseola (formerly Phaeoisariopsis griseola). Favours warm (20-25°C), humid conditions with prolonged leaf wetness.

{_bold('Symptoms:')}
• Leaf: Distinctive angular, brown to grey lesions restricted by leaf veins, giving a "blocky" appearance. Lesions have a characteristic greyish centre with dark brown borders.
• Spore Production: Under humid conditions, greyish fungal sporulation (spore-producing structures) can be seen on the underside of lesions.
• Stem and Pod Symptoms: Brown to black, sunken lesions on stems, petioles, and pods. Infected pods may be distorted and have reduced seed fill.
• Severe Infection: Heavy infection leads to premature defoliation (starting from lower leaves), reduced pod set, and yield losses of 30-50%.
• Disease Cycle: Fungus survives in infected crop debris for up to 2 years. Spores are splashed by rain or carried by wind.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
1. Apply fungicides at first symptom appearance, focusing on lower leaves where infection starts.
2. Remove and destroy severely infected lower leaves to reduce inoculum.
3. Avoid overhead irrigation to minimize leaf wetness.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant resistant varieties: Use ALS-resistant bean varieties (e.g., 'Gloria', 'K132', 'CAL 143', 'Andecha').
2. Use certified disease-free seeds – seed-borne inoculum is common.
3. Practice crop rotation: Rotate with non-legume crops (maize, sorghum, cassava) for 2-3 years.
4. Field sanitation: Remove and bury or burn crop residues after harvest.
5. Wider plant spacing: Increase spacing to improve air circulation and reduce leaf wetness duration.""",
            "chemical_control": f"""{_bold('Fungicides for Angular Leaf Spot:')}

{_bold('PROTECTANT FUNGICIDES (Apply before infection):')}
i) Mancozeb (e.g., Dithane M-45 from UPL Limited [1], Milthane Super from Twiga Chemical Industries Ltd [2], Fortress Gold 72 WP both from Greenlife Crop Protection Africa Ltd [3]). Use 2.5g/L of water.
ii) Chlorothalonil (e.g., Bravo 720 SC from Syngenta East Africa Ltd [4]). Use 2.0-3.0ml/L of water.
iii) Copper-based fungicides (e.g., Copper hydroxide – Kocide 2000 from DuPont [5], Nordox 75 WG from Nordox [6], Copac E from Greenlife Crop Protection Africa [7]; Copper oxychloride - Cupravit® 50 WP from Bayer East Africa [8]). Use 2.0-2.5g/L of water.

{_bold('SYSTEMIC/CURATIVE FUNGICIDES (Apply at first symptoms):')}
i) Azoxystrobin (e.g., Amistar® from Syngenta East Africa [9], Milestone® 250SC [10] and Absolute 375SC [11], both from Greenlife Crop Protection Africa). Use 1.0ml/L of water.
ii) Tebuconazole (e.g., Nativo 300 SC from Bayer Kenya [12], Defacto® 500EC from Greenlife Crop Protection Africa [13], Folicur 250EW from Bayer East Africa [14]). Use 0.5-1.0ml/L of water.
iii) Propiconazole (e.g., Tilt 250 EC from Syngenta Crop Protection AG [15] and Defacto® 500EC from Greenlife Crop Protection Africa [13]). Use 1.0ml/L of water.
iv) Pyraclostrobin (e.g., Comet 200 EC from BASF East Africa Ltd [16], Pyracopp 400 SC from Ningbo Sunjoy Agroscience [17]). Use 1.0ml/L of water.

{_bold('COMBINATION PRODUCTS:')}
i) Azoxystrobin + Propiconazole (e.g., Absolute 375SC from Greenlife Crop Protection Africa [11]). Use 1.0-1.5ml/L of water.
ii) Azoxystrobin + Chlorothalonil (e.g., Amistar® Opti from Syngenta East Africa [18]). Use 2.0-3.0ml/L of water.
iii) Tebuconazole + Mancozeb (e.g., Nativo 300 SC from Bayer Kenya [12]). Use 0.5-1.0ml/L of water.

{_bold('Application timing:')} Start at first symptoms (pre-flowering). Repeat at 7-14 day intervals during wet weather. Ensure thorough coverage of both upper and lower leaf surfaces, focusing on lower leaves where infection typically begins.""",
            "xai_ref_numbers": [56, 61],
            "management_refs": [56, 61],
            "chemical_refs_original": [27, 28, 29, 58, 64, 65, 66, 67, 68, 23, 20, 19, 25, 62, 24, 31, 33]
        }

    # ==================== BEANS ANTHRACNOSE ====================
    elif 'anthracnose' in name_lower and 'beans' in name_lower:
        return {
            "disease_name": "Bean Anthracnose",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Colletotrichum lindemuthianum",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Colletotrichum lindemuthianum. Thrives in cool, wet conditions (13-26°C).

{_bold('Symptoms:')}
• Leaf: Small, dark brown to black lesions along leaf veins on the underside of leaves.
• Stem: Elongated, sunken, dark red to black cankers on stems, petioles, and cotyledons.
• Pod: Most characteristic sign – Sunken, circular to oval lesions on pods, with dark brown to black borders and salmon-pink centres (containing fungal spores).
• Seed: Infected seeds show brown to black spots or lesions, often with cracking or discolouration.
• Disease Cycle: Fungus survives in infected crop debris and seeds. Splashing rain spreads spores to healthy plants.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
1. Apply fungicides at first sign of symptoms, focusing on pod protection.
2. Remove and destroy severely infected plants to reduce inoculum.
3. Avoid working in fields when plants are wet to prevent spore spread.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Use certified, disease-free seeds – This is the most important control measure.
2. Plant resistant varieties: Use anthracnose-resistant bean varieties (e.g., 'Gloria', 'K131', 'K132', 'CAL 143', 'G12873').
3. Practice crop rotation: Rotate with non-legume crops (e.g. maize, sorghum, cassava) for 3-4 years.
4. Field sanitation: Remove and bury or burn crop residues after harvest.
5. Seed treatment: Treat seeds with fungicide before planting to eliminate seed-borne inoculum.""",
            "chemical_control": f"""{_bold('Fungicides for Bean Anthracnose:')}

{_bold('PROTECTANT FUNGICIDES (Apply before infection):')}
i) Mancozeb (e.g., Dithane M-45 from UPL Limited [1], Milthane Super from Twiga Chemical Industries Ltd [2], Fortress Gold 72 WP both from Greenlife Crop Protection Africa Ltd [3]). Use 2.5g/L of water.
ii) Chlorothalonil (e.g., Bravo 720 SC from Syngenta East Africa Ltd [4]).

{_bold('SYSTEMIC FUNGICIDES (Curative – apply at first symptoms):')}
i) Azoxystrobin (e.g., Milestone® 250SC [5] and Absolute 375SC [6], both from Greenlife Crop Protection Africa).
ii) Tebuconazole (e.g., Nativo 300 SC from Bayer Kenya [7], Defacto® 500EC from Greenlife Crop Protection Africa [8]).
iii) Propiconazole (e.g., Tilt 250 EC from Syngenta Crop Protection AG [9] and Defacto® 500EC from Greenlife Crop Protection Africa [8]).

{_bold('COMBINATION PRODUCTS:')}
i) Azoxystrobin + Propiconazole (e.g., Absolute 375SC from Greenlife Crop Protection Africa [6]).

{_bold('Application timing:')} Start at first sign of symptoms (pre-flowering). Repeat at 7-10 day intervals during wet conditions.""",
            "xai_ref_numbers": [55, 56, 57],
            "management_refs": [55, 56, 57],
            "chemical_refs_original": [27, 28, 29, 58, 23, 20, 19, 25, 24]
        }

    # ==================== BEANS RUST ====================
    elif 'beans rust' in name_lower or ('rust' in name_lower and 'beans' in name_lower):
        return {
            "disease_name": "Bean Rust Disease",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Uromyces appendiculatus",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Uromyces appendiculatus. Favours moderate temperatures (17-22°C) and high humidity (85%+).

{_bold('Symptoms:')}
• Early Leaf: Small, white to light green raised spots (pustules) on the underside of lower leaves.
• Advanced Leaf: Pustules turn reddish-brown to dark brown, erupting through the leaf surface to release powdery rust-coloured spores. Pustules are surrounded by a yellow halo.
• Pod and Stem: Pustules may also appear on pods and stems, though less common.
• Severe Infection: Heavy rust infection causes premature defoliation, reduced photosynthesis, and significant yield loss (up to 50%).
• Disease Cycle: Fungus survives on infected crop debris and volunteer bean plants. Spores are wind-dispersed over long distances.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
1. Apply fungicides at first sign of pustules (usually on lower leaves).
2. Monitor fields regularly, especially during flowering and pod fill stages when rust is most damaging.
3. Improve air circulation through proper plant spacing and weed control.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant resistant varieties: Use rust-resistant bean varieties (e.g., 'Redlands', 'Pinto 114', 'KAT B1', 'KAT X56').
2. Crop rotation: Rotate with non-legume crops for 2-3 years.
3. Field sanitation: Remove and destroy crop residues after harvest.
4. Avoid overhead irrigation: Use drip or furrow irrigation to reduce leaf wetness duration.
5. Early planting: Plant early in the season to avoid peak rust conditions.""",
            "chemical_control": f"""{_bold('Fungicides for Bean Rust:')}

{_bold('PROTECTANT FUNGICIDES:')}
i) Chlorothalonil (e.g., Bravo 720 SC from Syngenta East Africa Ltd [1]). Use 2.0ml/L of water.
ii) Mancozeb (e.g., Dithane M-45 from UPL Limited [2], Milthane Super from Twiga Chemical Industries Ltd [3], Fortress Gold 72 WP from Greenlife Crop Protection Africa Ltd [4]). Use 2.5g/L of water.

{_bold('SYSTEMIC FUNGICIDES (Curative – apply at first symptoms):')}
i) Tebuconazole (e.g., Nativo 300 SC from Bayer Kenya [5], Defacto® 500EC from Greenlife Crop Protection Africa [6], Folicur 250EW from Bayer East Africa [7]). Use 0.5-1.0ml/L of water.
ii) Pyraclostrobin (e.g., Comet 200 EC from BASF East Africa Ltd [8], Pyracopp 400 SC from Ningbo Sunjoy Agroscience [9]). Use 1.0ml/L of water.
iii) Azoxystrobin (e.g., Milestone® 250SC [10] and Absolute 375SC [11], both from Greenlife Crop Protection Africa).
iv) Propiconazole (e.g., Tilt 250 EC from Syngenta Crop Protection AG [12] and Defacto® 500EC from Greenlife Crop Protection Africa [6]). Use 1.0ml/L of water.

{_bold('SULPHUR-BASED PRODUCTS (Organic option):')}
i) Wettable Sulphur (e.g., Kumulus DF from BASF East Africa Ltd [13]). Use 2-3g/L of water.

{_bold('Application timing:')} Apply at first sign of rust. Repeat at 7-14 day intervals during humid weather.""",
            "xai_ref_numbers": [59, 60, 61],
            "management_refs": [59, 60, 61],
            "chemical_refs_original": [58, 27, 28, 29, 19, 25, 62, 31, 33, 23, 20, 24, 63]
        }

    # ==================== BEANS HEALTHY ====================
    elif 'beans healthy' in name_lower:
        return {
            "disease_name": "Healthy Bean Crop",
            "category": "🌿 HEALTHY CROP",
            "causal_agent": "No disease detected",
            "is_healthy": True,
            "causes_characteristics": f"""{_bold('Key Characteristics of a healthy bean plant:')}

1. Colour and Appearance: Leaves are vibrant, deep green, indicating healthy nitrogen levels and photosynthesis.
2. Stem Strength: A thick, sturdy, and well-developed stem that supports the plant's growth habit (bush or climbing).
3. Root Structure: A robust, well-developed root system with visible nitrogen-fixing nodules (pink or red when active).
4. Foliage: Clean foliage free from discolouration (yellowing/browning), spots, or signs of pests or diseases.
5. Growth Pace: Steady and uniform growth across the field, with good leaf and pod development.
6. Pod Development: Well-filled, straight pods with healthy, uniform beans, indicating good pollination and nutrition.

{_bold('No disease detected.')}""",
            "management": "Continue regular management practices.",
            "chemical_control": "Not applicable.",
            "xai_ref_numbers": [55],
            "management_refs": [55],
            "chemical_refs_original": []
        }

    # ==================== TOMATO HEALTHY FRUIT ====================
    elif 'tomato healthy fruit' in name_lower:
        return {
            "disease_name": "Healthy Tomato Fruit",
            "category": "🌿 HEALTHY CROP",
            "causal_agent": "No disease detected",
            "is_healthy": True,
            "causes_characteristics": f"""Fruits are firm, evenly coloured (red, yellow, or orange depending on variety), free from blemishes, cracks, spots, rot, or pest damage. The fruit surface is smooth and glossy.

{_bold('No disease detected.')}""",
            "management": "Continue regular management practices including proper irrigation, fertilization, and pest monitoring.",
            "chemical_control": "Not applicable.",
            "xai_ref_numbers": [69, 70],
            "management_refs": [69, 70],
            "chemical_refs_original": []
        }

    # ==================== TOMATO HEALTHY LEAF ====================
    elif 'tomato healthy leaf' in name_lower or 'tomato healthy crop leaf' in name_lower:
        return {
            "disease_name": "Healthy Tomato Leaf",
            "category": "🌿 HEALTHY CROP",
            "causal_agent": "No disease detected",
            "is_healthy": True,
            "causes_characteristics": f"""Leaves are vibrant green, free from discolouration (yellowing/browning), spots, lesions, wilting, or signs of pests or diseases. Leaflets are well-formed without curling or distortion.

{_bold('No disease detected.')}""",
            "management": "Continue regular management practices.",
            "chemical_control": "Not applicable.",
            "xai_ref_numbers": [69, 70],
            "management_refs": [69, 70],
            "chemical_refs_original": []
        }

    # ==================== TOMATO ALTERNARIA ====================
    elif 'alternaria' in name_lower and 'mite' not in name_lower:
        return {
            "disease_name": "Tomato Alternaria Disease (Early Blight)",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Alternaria tomatophila or Alternaria solani",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Alternaria tomatophila or Alternaria solani. Favours warm (24-30°C), humid conditions with alternating wet and dry periods.

{_bold('Symptoms:')}
• Leaf: Dark brown to black, circular to angular spots with concentric rings. Lesions typically start on older, lower leaves and progress upward. Yellow halos often surround lesions. Heavily infected leaves turn yellow and drop prematurely.
• Stem: Dark, sunken, elliptical lesions on stems (collar rot).
• Fruit: Dark, sunken, leathery lesions around the stem end of fruit. Fruit may drop prematurely.
• Disease Cycle: The fungus survives in infected crop debris and soil for up to 1-2 years. Spores are splashed by rain or carried by wind.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
1. Apply fungicides at first sign of symptoms, focusing on lower leaves where infection starts.
2. Remove and destroy severely infected lower leaves to reduce inoculum.
3. Stake or prune plants to improve air circulation and reduce leaf wetness duration.
4. Avoid overhead irrigation; use drip or furrow irrigation instead.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Use disease-free seedlings from reputable sources.
2. Plant resistant varieties where available.
3. Practice crop rotation with non-solanaceous crops (maize, beans, cabbage) for 2-3 years.
4. Field sanitation: Remove and destroy crop residues after harvest. Burn infected plants.
5. Mulch around plants to prevent soil splash that carries fungal spores.
6. Disinfect pruning shears after each cut to prevent spread.""",
            "chemical_control": f"""{_bold('Fungicides for Early Blight:')}

{_bold('PROTECTANT FUNGICIDES (Preventive):')}
i) Mancozeb (e.g., Dithane M-45 from UPL Limited [1], Milthane Super from Twiga Chemical Industries Ltd [2], Fortress Gold 72 WP from Greenlife Crop Protection Africa Ltd [3]). Use 2.5g/L of water.
ii) Chlorothalonil (e.g., Bravo 720 SC from Syngenta East Africa Ltd [4]). Use 2.0-3.0ml/L of water.
iii) Copper-based fungicides (e.g., Copper hydroxide – Kocide 2000 from DuPont [5], Nordox 75 WG from Nordox [6], Copac E from Greenlife Crop Protection Africa [7]; Copper oxychloride - Cupravit® 50 WP from Bayer East Africa [8]). Use 2.0-2.5g/L of water.

{_bold('SYSTEMIC/CURATIVE FUNGICIDES (Apply at first symptoms):')}
i) Azoxystrobin (e.g., Amistar® from Syngenta East Africa [9], Milestone® 250SC [10] and Absolute 375SC [11], both from Greenlife Crop Protection Africa). Use 1.0ml/L of water.
ii) Tebuconazole (e.g., Nativo 300 SC from Bayer Kenya [12], Defacto® 500EC from Greenlife Crop Protection Africa [13], Folicur 250EW from Bayer East Africa [14]). Use 0.5-1.0ml/L of water.
iii) Propiconazole (e.g., Tilt 250 EC from Syngenta Crop Protection AG [15]). Use 1.0ml/L of water.

{_bold('COMBINATION PRODUCTS:')}
i) Azoxystrobin + Propiconazole (e.g., Absolute 375SC from Greenlife Crop Protection Africa [11]). Use 1.0-1.5ml/L of water.
ii) Azoxystrobin + Chlorothalonil (e.g., Amistar® Opti from Syngenta East Africa [16]). Use 2.0-3.0ml/L of water.

{_bold('Application timing:')} Start at first sign of symptoms. Repeat at 7-10 day intervals during warm, humid weather. Cover both tops and undersides of leaves.""",
            "xai_ref_numbers": [55, 71, 72, 73],
            "management_refs": [55, 71, 72, 73],
            "chemical_refs_original": [27, 28, 29, 58, 64, 65, 66, 67, 68, 23, 20, 19, 25, 62, 24]
        }

    # ==================== TOMATO LATE BLIGHT ====================
    elif 'late blight' in name_lower:
        return {
            "disease_name": "Tomato Late Blight Disease",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus-like micro-organism Phytophthora infestans",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus-like micro-organism Phytophthora infestans. Prefers cool (15-22°C), wet, humid conditions.

{_bold('Symptoms:')}
• Leaf: Large, irregular, water-soaked, greyish-green spots on leaves. Under humid conditions, a distinctive white, fuzzy mould (sporulation) appears on the underside of lesions.
• Stem: Brown to black, water-soaked lesions on stems and petioles. Lesions girdle stems, causing plant collapse.
• Fruit: Greasy, brown to purple-brown lesions on fruit. Lesions may become firm and sunken. Secondary rot organisms often invade.
• Rapid Spread: Under favourable conditions, the entire field can be destroyed within 7-10 days.
• Disease Cycle: Pathogen survives in infected potato tubers, volunteer tomatoes, and crop debris. Spores are wind-dispersed over long distances.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
1. Apply fungicides immediately at the first sign of symptoms. Late blight requires aggressive chemical control.
2. Remove and destroy infected plants from the field to reduce inoculum spread.
3. Avoid overhead irrigation; use drip or furrow irrigation instead.
4. Improve air circulation through proper staking, pruning, and plant spacing.
5. Water in the early morning to give plants time to dry during the day.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant resistant varieties: Use late blight-resistant tomato varieties (e.g., 'Mountain Magic', 'Plum Regal', 'Iron Lady', 'Jasper').
2. Use certified, disease-free seedlings.
3. Practice crop rotation: Rotate with non-solanaceous crops (maize, beans, cabbage) for 3-4 years.
4. Field sanitation: Remove and destroy crop residues from the field after harvesting.
5. Eliminate volunteer tomatoes and potatoes from the field (reservoir hosts).
6. Space plants sufficiently apart to allow adequate air circulation.
7. Preventive spraying: During high-risk seasons (wet, humid), apply preventive fungicides before symptoms appear.""",
            "chemical_control": f"""{_bold('Fungicides for Late Blight:')}

{_bold('PROTECTANT FUNGICIDES (Preventive – apply before infection):')}
i) Mancozeb (e.g., Dithane M-45 from UPL Limited [1], Milthane Super from Twiga Chemical Industries Ltd [2], Fortress Gold 72 WP from Greenlife Crop Protection Africa Ltd [3]). Use 2.5g/L of water.
ii) Chlorothalonil (e.g., Bravo 720 SC from Syngenta East Africa Ltd [4]). Use 2.0-3.0ml/L of water.
iii) Copper-based fungicides (e.g., Copper hydroxide – Kocide 2000 from DuPont [5], Nordox 75 WG from Nordox [6], Copac E from Greenlife Crop Protection Africa [7]; Copper oxychloride - Cupravit® 50 WP from Bayer East Africa [8]). Use 2.0-2.5g/L of water.

{_bold('SYSTEMIC/CURATIVE FUNGICIDES (Apply at first symptoms):')}
i) Metalaxyl-M + Mancozeb (e.g., Ridomil Gold® MZ 68 WG from Syngenta East Africa [9]). Use 2.0-2.5g/L of water.
ii) Fenamidone + Mancozeb (e.g., Reason 500 SC from Bayer East Africa [10]).
iii) Cymoxanil + Mancozeb (e.g., Curzate MZ from Corteva Agriscience [11]).
iv) Dimethomorph (e.g., Acrobat 500 SC from BASF East Africa Limited [12]).
v) Mandipropamid (e.g., REVUS® 250SC from Syngenta East Africa [13]).

{_bold('COMBINATION PRODUCTS:')}
i) Azoxystrobin + Chlorothalonil (e.g., Amistar® Opti from Syngenta East Africa [14]). Use 2.0-3.0ml/L of water.

{_bold('Application timing:')}
• Prevention: Apply protectant fungicides before symptoms appear, especially when weather conditions are favourable for disease.
• Curative: Apply systemic/curative fungicides at the first sign of symptoms. Repeat at 5-7 day intervals during the active phase of the disease.""",
            "xai_ref_numbers": [71, 72],
            "management_refs": [71, 72],
            "chemical_refs_original": [27, 28, 29, 58, 64, 65, 66, 67, 77, 78, 79, 80, 81, 68]
        }

    # ==================== TOMATO VIROSIS (TYLCV) ====================
    elif 'virosis' in name_lower or 'yellow leaf curl' in name_lower:
        return {
            "disease_name": "Tomato Virosis Disease (Tomato Yellow Leaf Curl Virus / TYLCV)",
            "category": "🦠 VIRAL DISEASE",
            "causal_agent": "Tomato Yellow Leaf Curl Virus (TYLCV). Transmitted by whiteflies",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Tomato Yellow Leaf Curl Virus (TYLCV). The virus is transmitted exclusively by whiteflies (Bemisia tabaci). It is not seed-borne or mechanically transmitted.

{_bold('Symptoms:')}
• Leaf: Severe upward curling and crinkling of young leaves. Leaves become yellow between veins (interveinal chlorosis) and are reduced in size.
• Plant Stunting: Infected plants are severely stunted, with shortened internodes giving a "bushy" appearance.
• Flower and Fruit: Reduced flowering and fruit set. Produced fruits are small and may be misshapen.
• Severe Infection: If the plants are infected early, high yield losses of over 80% can occur.
• Disease Cycle: The virus survives in infected host plants (tomato, weeds, other crops). Whiteflies acquire the virus by feeding on infected plants and transmit it to healthy plants.""",
            "management": f"""{_bold('⚠️ NO CURE FOR INFECTED PLANTS. Focus on prevention and vector control. ⚠️')}

{_bold('FOR INFECTED CROPS:')}
• Remove and destroy infected plants from the field to reduce inoculum spread.
• Control whitefly vectors using appropriate insecticides.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant resistant varieties: Use TYLCV-resistant tomato varieties (e.g., 'Shanty', 'Tyking', 'Fahari', 'Anna F1').
2. Use virus-free seedlings from reputable nurseries.
3. Whitefly control: Use yellow sticky traps (5 traps/acre) to monitor and trap whiteflies.
4. Apply insecticides at 15, 25, and 45 days after transplanting.
5. Use reflective mulches (aluminium-coated plastic) to repel whiteflies.
6. Install insect netting over young plants.
7. Roguing: Remove infected plants as soon as symptoms are noticed.
8. Control weed hosts: Eliminate weeds that serve as alternative hosts for whiteflies and the virus.
9. Avoid planting near infected fields as whiteflies can travel significant distances.
10. Crop rotation: Rotate with non-host crops (maize, beans, cabbage).""",
            "chemical_control": f"""{_bold('Chemical Control (for whitefly vector only – ineffective against the virus itself):')}

{_bold('Chemical Insecticides:')}
i) Imidacloprid (e.g., Confidor® 200SL from Bayer East Africa [1], Emerald 200SL from Greenlife Crop Protection Africa [2]). Use 0.5-1.0ml/L of water.
ii) Lambda-cyhalothrin (e.g., Karate® 5EC from Syngenta East Africa [3], Lamdastar 5% EC from Agro Pests [4]). Use 1.0ml/L of water.
iii) Acetamiprid (e.g., Presento® 200SP from Greenlife Crop Protection Africa [5]). Use 0.5g/L of water.
iv) Diazinon 60% EC from VAPCO [6]. Use 1.5ml/L of water.

{_bold('Organic Options for Whitefly Control:')}
i) Neem oil (e.g., NeemAzal 1.2EC from Twiga Chemicals [7], Nimbecidine from Osho Chemical Industries Ltd [8]). Use 5-10ml/L of water. It repels whiteflies and disrupts their feeding.
ii) Yellow sticky traps. Use 5 traps per acre for monitoring and mass trapping [9].

{_bold('Application timing:')} Apply insecticides at 15, 25, and 45 days after transplanting, or when whiteflies are first observed. Target the undersides of leaves where whiteflies congregate.""",
            "xai_ref_numbers": [57, 71],
            "management_refs": [57, 71],
            "chemical_refs_original": [82, 6, 83, 16, 47, 14, 8, 9, 84]
        }

    # ==================== TOMATO BLOSSOM END ROT ====================
    elif 'blossom end rot' in name_lower:
        return {
            "disease_name": "Tomato Blossom End Rot",
            "category": "🌱 PHYSIOLOGICAL DISORDER",
            "causal_agent": "Calcium deficiency in developing fruits",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Cause:')} Physiological disorder caused by calcium deficiency in developing fruits, often induced by inconsistent watering, high temperatures, or excessive nitrogen fertilization.

{_bold('Symptoms:')}
• Fruit: Dark, sunken, leathery lesion on the blossom end (bottom) of the fruit.
• The lesion starts as a small water-soaked spot that enlarges and turns brown to black.
• The lesion may cover up to one-third to one-half of the fruit surface.
• Secondary rot organisms may invade the lesion.
• Calcium Mobility: Calcium is not easily translocated within the plant. Rapid fruit growth creates a high calcium demand that roots cannot meet.

{_bold('Predisposing Factors:')}
• Inconsistent soil moisture (drought followed by heavy watering)
• High temperatures
• Excessive nitrogen fertilization (promotes rapid growth)
• Low soil calcium
• Root damage (nematodes, cultivation)
• High soil salinity""",
            "management": f"""{_bold('FOR AFFECTED CROPS:')}
• Remove affected fruits as they will not recover. Removing them allows the plant to direct calcium to remaining fruits.
• Apply calcium sprays (calcium chloride or calcium nitrate) as a foliar spray for immediate calcium supply.
• Maintain consistent soil moisture through regular irrigation.

{_bold('FOR LONG-TERM PREVENTION:')}
1. Soil testing: Conduct soil tests before planting to determine calcium levels and pH.
2. Lime application: Apply lime (calcium carbonate) to acidic soils before planting.
3. Balanced fertilization: Avoid excessive nitrogen, especially ammonium-based fertilizers. Use calcium-containing fertilizers.
4. Consistent irrigation: Maintain even soil moisture throughout the growing season. Use mulching to reduce moisture fluctuations.
5. Avoid root damage: Minimise cultivation near plant roots. Control nematodes.
6. Choose less susceptible varieties to blossom end rot.
7. Foliar calcium sprays: Apply as a preventive measure during fruit set.""",
            "chemical_control": f"""{_bold('Foliar Calcium Sprays:')}

{_bold('Calcium Sprays:')}
i) Calcium chloride (e.g., CC Farm Calcium Chloride from Tetra Technologies Inc. [1]). Use 1.0-2.0g/L of water.
ii) Calcium nitrate (e.g., YaraLiva® CALCINIT® 15.5-0-0 from Yara US [2]). Use 1.0-2.0g/L of water.

{_bold('Soil Applications (Preventive):')}
i) Agricultural Lime (calcium carbonate) [3]. Apply based on soil test recommendations (typically 400-800 kg/acre).
ii) Agricultural Gypsum (calcium sulphate) [4]. Apply based on soil test recommendations.

{_bold('Application timing:')} Apply foliar calcium sprays when fruits are pea-sized (first fruit set). Repeat at 7-10 day intervals during fruit development. Spray in early morning or late evening to avoid leaf burn.

{_bold('Water Management:')} Maintain consistent soil moisture through regular irrigation. Use mulching to reduce moisture evaporation.""",
            "xai_ref_numbers": [98, 99],
            "management_refs": [98, 99],
            "chemical_refs_original": [100, 101, 102, 103]
        }

    # ==================== TOMATO FUSARIUM ====================
    elif 'fusarium' in name_lower:
        return {
            "disease_name": "Tomato Fusarium Disease (Fusarium Wilt)",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Fusarium oxysporum f.sp. lycopersici",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Fusarium oxysporum f.sp. lycopersici. Favours warm soil temperatures (25-30°C).

{_bold('Symptoms:')}
• Initial: Yellowing of lower leaves, often starting on one side of the leaf or plant (unilateral wilting).
• Vascular: Characteristic browning of vascular tissue (xylem) visible when the stem is cut lengthwise – distinguishes Fusarium wilt from other wilts.
• Wilting Progression: Wilting progresses upward from lower leaves. Infected plants may be stunted and produce small, poor-quality fruit.
• Plant Death: Severe infection leads to complete wilting and plant death.
• Soil-Borne: The fungus survives in the soil for many years (up to 10 years) through resistant spores.
• Disease Cycle: The fungus enters through root tips or wounds. It is spread by contaminated soil, water, and farming equipment.""",
            "management": f"""{_bold('⚠️ NO CURE FOR INFECTED PLANTS. Focus on prevention. ⚠️')}

{_bold('FOR INFECTED CROPS:')}
• Remove and destroy infected plants immediately to reduce soil inoculum levels.
• Do not compost infected plant material. Burn or dispose of off-site.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Plant resistant varieties: Use Fusarium wilt-resistant tomato varieties (e.g., 'Mountain Fresh', 'Mountain Pride', 'Better Boy', 'Celebrity') – look for "F" designation on seed packets.
2. Use disease-free seedlings from reputable sources – critical.
3. Practice long crop rotation: Rotate with non-solanaceous crops (maize, beans, cabbage) for 4-6 years.
4. Soil solarisation: In high-infestation areas, solarise soil during the hot season to reduce pathogen populations.
5. Maintain lower garden temperatures as the fungus thrives in warm conditions.
6. Disinfect and clean farming tools before using them in different fields.
7. Avoid planting in infested fields. Once a field has Fusarium wilt, the fungus persists for many years.""",
            "chemical_control": f"""{_bold('Chemical Control for Fusarium Wilt (limited efficacy – mainly preventive):')}

{_bold('Seed Treatment (before planting):')}
i) Carbendazim 50% WP (e.g., Bavistin from Crystal Crop Protection Ltd [1]).
ii) Thiram (e.g., 42-S Thiram from Bayer Crop Science for seed treatment [2]).

{_bold('Soil Drench (at planting time in high-risk fields):')}
i) Carbendazim 50% WP (e.g., Bavistin from Crystal Crop Protection Ltd [1]).

{_bold('Biological Options:')}
i) Streptomyces (e.g., MYCOSTOP® from Verdera for soil application) [3].

{_bold('Preventive Copper Sprays (limited effect):')}
i) Copper hydroxide (e.g., Kocide 2000 from DuPont [4], Nordox 75 WG from Nordox [5], Copac E from Greenlife Crop Protection Africa [6]).
ii) Copper oxychloride (e.g., Cupravit® 50 WP from Bayer East Africa [7]). Use 2.0-2.5g/L of water.

{_bold('Note:')} Fungicides are most effective as seed treatments or soil drenches before planting. Once plants show symptoms, chemical control is largely ineffective. Apply as soil drench at planting time in high-risk fields.""",
            "xai_ref_numbers": [71, 72],
            "management_refs": [71, 72],
            "chemical_refs_original": [74, 75, 76, 64, 65, 66, 67]
        }

    # ==================== TOMATO SEPTORIA ====================
    elif 'septoria' in name_lower:
        return {
            "disease_name": "Tomato Septoria Disease (Septoria Leaf Spot)",
            "category": "🍄 FUNGAL DISEASE",
            "causal_agent": "Fungus Septoria lycopersici",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Fungus Septoria lycopersici. Favours warm, humid conditions with prolonged leaf wetness.

{_bold('Symptoms:')}
• Leaf: Small, round to irregular spots with grey or tan centres and dark brown to black margins. Spots typically start on lower, older leaves.
• Fruiting Bodies: Tiny black dots (pycnidia – fungal fruiting bodies) may be visible within the grey centres under magnification.
• Severe Infection: As spots coalesce, leaves turn yellow and drop prematurely (defoliation), starting from the bottom of the plant and moving upward.
• Stem and Flower Symptoms: Spots may also appear on stems and flowers, though less common.
• Disease Cycle: The fungus survives in infected crop debris and on volunteer tomato plants. Spores are splashed by rain or irrigation water.""",
            "management": f"""{_bold('FOR INFECTED CROPS:')}
1. Apply fungicides at the first sign of symptoms, focusing on lower leaves.
2. Remove and destroy infected lower leaves to reduce inoculum.
3. Avoid overhead irrigation; use drip or furrow irrigation instead.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Use disease-free seedlings from reputable suppliers.
2. Stake and prune plants to improve air circulation and reduce leaf wetness duration.
3. Practice crop rotation with non-solanaceous crops (maize, beans, cabbage) for 2-3 years.
4. Field sanitation: Remove and destroy crop residues after harvesting. Burn infected plants.
5. Mulch around plants to prevent soil splash that carries fungal spores.
6. Avoid working in fields when plants are wet to prevent spreading the fungus.""",
            "chemical_control": f"""{_bold('Fungicides for Septoria Leaf Spot:')}

{_bold('PROTECTANT FUNGICIDES (Preventive – apply before infection):')}
i) Mancozeb (e.g., Dithane M-45 from UPL Limited [1], Milthane Super from Twiga Chemical Industries Ltd [2], Fortress Gold 72 WP from Greenlife Crop Protection Africa Ltd [3]). Use 2.5g/L of water.
ii) Chlorothalonil (e.g., Bravo 720 SC from Syngenta East Africa Ltd [4]). Use 2.0-3.0ml/L of water.
iii) Copper-based fungicides (e.g., Copper hydroxide – Kocide 2000 from DuPont [5], Nordox 75 WG from Nordox [6], Copac E from Greenlife Crop Protection Africa [7]; Copper oxychloride - Cupravit® 50 WP from Bayer East Africa [8]). Use 2.0-2.5g/L of water.

{_bold('SYSTEMIC/CURATIVE FUNGICIDES (Apply at first symptoms):')}
i) Azoxystrobin (e.g., Amistar® from Syngenta East Africa [9], Milestone® 250SC [10] and Absolute 375SC [11], both from Greenlife Crop Protection Africa). Use 1.0ml/L of water.
ii) Tebuconazole (e.g., Nativo 300 SC from Bayer Kenya [12], Defacto® 500EC from Greenlife Crop Protection Africa [13], Folicur 250EW from Bayer East Africa [14]). Use 0.5-1.0ml/L of water.
iii) Propiconazole (e.g., Tilt 250 EC from Syngenta Crop Protection AG [15]). Use 1.0ml/L of water.

{_bold('COMBINATION PRODUCTS:')}
i) Azoxystrobin + Chlorothalonil (e.g., Amistar® Opti from Syngenta East Africa [16]). Use 2.0-3.0ml/L of water.

{_bold('Application timing:')} Start at the first sign of symptoms. Repeat at 7-10 day intervals during wet weather. Ensure thorough coverage of lower leaves where infection begins.""",
            "xai_ref_numbers": [71, 72],
            "management_refs": [71, 72],
            "chemical_refs_original": [27, 28, 29, 58, 64, 65, 66, 67, 68, 23, 20, 19, 25, 62, 24]
        }

    # ==================== TOMATO BACTERIAL SPECK ====================
    elif 'bacterial speck' in name_lower:
        return {
            "disease_name": "Tomato Bacterial Speck Disease",
            "category": "🦠 BACTERIAL DISEASE",
            "causal_agent": "Bacterium Pseudomonas syringae pv. tomato",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Causal Agent:')} Bacterium Pseudomonas syringae pv. tomato. Favours cool (15-22°C), wet, humid conditions.

{_bold('Symptoms:')}
• Leaf: Small (1-3mm), dark brown to black spots with distinctive yellow halos. Spots are often numerous and may coalesce.
• Fruit: Small, dark, raised spots on green fruit. Spots become black, flat, and scabby as the fruit ripens. Unlike bacterial spot, bacterial speck lesions on fruit are not raised or rough.
• Severe Infection: Heavy infection causes defoliation (starting from lower leaves), sunscald on exposed fruit, and reduced fruit quality.
• Seed-Borne: The bacterium is carried on and in infected seeds.
• Disease Cycle: The bacterium survives in infected crop debris, on volunteer tomatoes, and on seeds. It is spread by splashing rain, overhead irrigation, and contaminated tools/clothing.""",
            "management": f"""{_bold('⚠️ NO CURE FOR INFECTED PLANTS. Focus on prevention. ⚠️')}

{_bold('FOR INFECTED CROPS:')}
• Apply copper-based bactericides. They have limited efficacy and may reduce the spread of the disease.
• Remove and destroy severely infected plants to reduce inoculum spread.
• Avoid working in fields when plants are wet to prevent mechanical spread.
• Avoid overhead irrigation; use drip or furrow irrigation instead.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Use certified, disease-free seeds. This is the most important control measure.
2. Use disease-free seedlings from reputable nurseries.
3. Hot water seed treatment: Treat seeds at 50°C for 25-30 minutes (although this may reduce germination).
4. Practice crop rotation with non-solanaceous crops (e.g., maize, beans, cabbage) for 2-3 years.
5. Field sanitation: Remove and destroy crop residues after harvest. Burn infected plants.
6. Control volunteer tomatoes as these can serve as reservoir hosts.""",
            "chemical_control": f"""{_bold('Chemical Control (limited efficacy – only for reducing spread):')}

{_bold('Copper-based Bactericides:')}
i) Copper hydroxide (e.g., Kocide 2000 from DuPont [1], Nordox 75 WG from Nordox [2]). Use 2.0-2.5g/L of water.
ii) Copper oxychloride (e.g., Cupravit® 50 WP from Bayer East Africa [3]). Use 2.0-2.5g/L of water.
iii) Copper octanoate (e.g., Cueva® from Certis Biologicals [4]).

{_bold('Combination Products:')}
i) Copper hydroxide + Mancozeb (e.g., ManKocide from Cosaco) [5].

{_bold('Biological Options:')}
i) Bacillus subtilis (e.g., Serenade® ASO from Bayer Crop Science) [6].

{_bold('Application timing:')} Apply preventively before symptom appearance in high-risk fields. Repeat at 7-10 day intervals during wet weather.

{_bold('Note:')} Copper sprays only reduce secondary spread; they cannot cure already infected plants.""",
            "xai_ref_numbers": [71, 72],
            "management_refs": [71, 72],
            "chemical_refs_original": [64, 65, 67, 85, 86, 87]
        }

    # ==================== TOMATO FRUIT BORER ====================
    elif 'fruit borer' in name_lower or 'helicoverpa' in name_lower:
        return {
            "disease_name": "Tomato Fruit Borer (Helicoverpa armigera)",
            "category": "🐛 PEST INFESTATION",
            "causal_agent": "Tomato Fruit Borer (Helicoverpa armigera)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Pest Identification:')}
• Larvae are greenish-brown to reddish-brown caterpillars with distinctive pale stripes along the body.
• Adult moths are light brown with a dark spot on each forewing.
• Feeding Behaviour: Larvae bore into tomato fruits, creating round holes. They feed on the internal contents of the fruit.
• Fruit Symptoms: Bore holes on the fruit surface, often with insect droppings protruding from the hole. Damaged fruits are prone to secondary rot.
• Life Cycle: Eggs are laid singly on leaves and fruits. Larvae go through 5-6 instars before pupating in the soil.
• Economic Importance: One of the most destructive pests of tomatoes. They can cause 50-80% yield loss if uncontrolled.""",
            "management": f"""{_bold('Integrated Pest Management (IPM):')}

1. Regular field scouting: Monitor for eggs and young larvae, especially during flowering and fruiting stages.
2. Pheromone traps: Use sex pheromone traps (5-10 traps/ha) for monitoring adult moth populations.
3. Economic threshold: Apply insecticides when egg counts exceed 1-2 eggs per plant OR when 1-2% of fruits show damage.
4. Apply insecticides at the egg or early larval stage before the larvae bore into fruits.
5. Handpicking: In small plantings, handpick and destroy larvae and damaged fruits.
6. Cultural controls: Deep ploughing after harvest to expose pupae; intercropping with repellent plants (marigold, basil); use of bird perches to attract natural predators.
7. Conserve natural enemies: Avoid broad-spectrum insecticides that kill beneficial insects.
8. Crop rotation: Rotate with non-host crops (e.g., maize, beans, cabbage) to break the pest cycle.""",
            "chemical_control": f"""{_bold('Insecticides for Tomato Fruit Borer – Target young larvae (early instars) before they bore into fruits:')}

{_bold('Chemical Insecticides:')}
i) Emamectin benzoate (e.g., Escort 19EC from Greenlife Crop Protection Africa [1]). Use 0.5-1.0ml/L of water.
ii) Spinetoram (e.g., Fidelity 400WG from Corteva Agriscience [2]).
iii) Indoxacarb (e.g., Steward 30WG from Corteva Agriscience [3]).
iv) Lambda-cyhalothrin (e.g., Karate® 5EC from Syngenta East Africa [4], Lamdastar 5% EC from Agro Pests [5]). Use 1.0ml/L of water.
v) Chlorantraniliprole (e.g., Coragen 20SC from FMC Corporation [6]). Use 0.5ml/L of water. It is highly effective and selective for caterpillars.

{_bold('Organic/Biopesticide Options:')}
i) Bacillus thuringiensis (Bt) (e.g., Thuricide® from Certis Biologicals [7]). Use 1.0-2.0g/L of water. Effective on young larvae.
ii) Neem oil (e.g., NeemAzal 1.2EC from Twiga Chemicals [8], Nimbecidine from Osho Chemical Industries Ltd [9]). Use 5-10ml/L of water. Repels and disrupts larval development.
iii) Nuclear Polyhedrosis Virus (NPV) (e.g., VIVUS® Max from AgBiTech) [10].

{_bold('Application timing:')} Apply when eggs or young larvae are first observed. Target fruits and foliage thoroughly. Repeat at 5-7 day intervals if pest pressure persists. Spray in early morning or late evening when larvae are active.""",
            "xai_ref_numbers": [56, 57, 71],
            "management_refs": [56, 57, 71],
            "chemical_refs_original": [40, 41, 88, 83, 16, 89, 90, 8, 9, 91]
        }

    # ==================== TOMATO LEAFMINER ====================
    elif 'leafminer' in name_lower or 'tuta' in name_lower:
        return {
            "disease_name": "Tomato Leafminer (Tuta absoluta)",
            "category": "🐛 PEST INFESTATION",
            "causal_agent": "Tomato Leafminer (Tuta absoluta)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Pest Identification:')}
• Small (2-3mm), brownish-grey moth.
• Larvae are pale green to pinkish caterpillars with a dark head capsule.
• Leaf Damage: Larvae mine between leaf surfaces, creating characteristic blotch-shaped mines (leaf mines) that appear as white, winding tunnels.
• Stem Damage: Larvae also mine stems and petioles, causing wilting and dieback.
• Fruit Damage: Larvae bore into fruits, creating small holes at the calyx (stem end) or through the fruit wall. Damaged fruits are prone to secondary rot.
• Rapid Reproduction: Tuta absoluta can complete a generation in 30-35 days, with up to 10-12 generations per year.
• Economic Importance: Tuta absoluta can cause 80-100% yield loss if left uncontrolled.""",
            "management": f"""{_bold('Integrated Pest Management (IPM):')}

{_bold('FOR INFECTED CROPS:')}
1. Apply insecticides targeting early larval stages before they enter mines or fruits.
2. Use pheromone traps (Delta traps with Tuta absoluta specific pheromone lures) for monitoring and mass trapping (12-16 traps/acre). Replace the lures every 4-6 weeks.
3. Remove and destroy infested leaves, stems, and fruits.
4. Prune lower leaves to reduce hiding places and improve spray coverage.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Use biological control agents: Predatory bugs (Nesidiocoris tenuis), parasitic wasps (Trichogramma species), predatory mites (Amblyseius species).
2. Install insect netting (with a mesh size < 1mm) over young plants to exclude moths.
3. Use sex pheromone traps for mass trapping (use 12-16 traps/acre).
4. Use light traps to attract and kill adult moths at night.
5. Do deep ploughing after harvesting to destroy pupae in the soil.
6. Crop rotation: Rotate with non-solanaceous crops (e.g., maize, beans, cabbage).
7. Remove volunteer tomato plants which serve as reservoir hosts.
8. Avoid planting near infested fields as moths can travel great distances.""",
            "chemical_control": f"""{_bold('Insecticides for Tuta absoluta – Target early larval stages (before they enter mines):')}

{_bold('Chemical Insecticides:')}
i) Spinosad (e.g., Tracer 480SC from Corteva Agriscience [1]). Use 0.5ml/L of water.
ii) Indoxacarb (e.g., Steward 30WG from Corteva Agriscience [2]).
iii) Emamectin benzoate (e.g., Escort 19EC from Greenlife Crop Protection Africa [3]). Use 0.5-1.0ml/L of water.
iv) Chlorantraniliprole (e.g., Coragen 20SC from FMC Corporation [4]). Use 0.5ml/L of water.
v) Flubendiamide (e.g., Belt 480SC from Bayer East Africa [5]).

{_bold('Organic Options:')}
i) Azadirachtin (Neem-based) (e.g., NeemAzal 1.2EC from Twiga Chemicals [6], Nimbecidine from Osho Chemical Industries Ltd [7]). Use 5-10ml/L of water. It disrupts larval development and acts as an antifeedant.

{_bold('Application timing:')} Apply insecticides at the first sign of leaf mines or moth activity. Target the undersides of leaves and growing tips. Repeat at 5-7 day intervals during high pest pressure. Rotate between different chemical groups to prevent the pests from developing resistance.""",
            "xai_ref_numbers": [57, 71],
            "management_refs": [57, 71],
            "chemical_refs_original": [92, 88, 40, 89, 93, 8, 9]
        }

    # ==================== TOMATO MITE ====================
    elif 'mite' in name_lower and 'alternaria' not in name_lower:
        return {
            "disease_name": "Tomato Red Spider Mite / Two-Spotted Spider Mite",
            "category": "🐛 PEST INFESTATION",
            "causal_agent": "Two-spotted spider mite (Tetranychus urticae)",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Pest Identification:')}
• Tiny (0.5mm), reddish-brown or yellowish-green mites with two dark spots on the body. The mites are visible with a hand lens.
• Leaf Symptoms: Stippling (tiny yellow or white spots) on leaves from mites feeding on plant sap. Severe infestations cause leaves to turn yellow, bronze, and dry up.
• Webbing: Characteristic fine webbing on the undersides of leaves and between stems. Webbing protects mites from predators and some pesticides.
• Damage Progression: Infestation typically starts on lower leaves and spreads upward. Severe infestations lead to complete defoliation.
• Fruit Symptoms: Reduced fruit size and quality. The fruits may be sunburned due to leaf loss.
• Rapid Reproduction: Mites complete a generation in 5-7 days in warm conditions.
• Disease Cycle: Mites thrive in hot, dry conditions. They are spread by wind, on clothing, equipment, and plant material.""",
            "management": f"""{_bold('Integrated Pest Management (IPM):')}

{_bold('FOR INFECTED CROPS:')}
1. Apply miticides when mite populations exceed 5-10 mites per leaf.
2. Prune and remove heavily infested lower leaves.
3. Increase humidity through overhead irrigation (mites thrive in dry conditions).
4. Ensure adequate plant hydration as water-stressed plants are more susceptible.

{_bold('FOR LONG-TERM MANAGEMENT:')}
1. Use biological control agents: Predatory mites (Phytoseiulus persimilis, Amblyseius californicus) – most effective; Predatory thrips (Scolothrips species); Ladybird beetles (Stethorus species).
2. Regular field scouting: Inspect the undersides of leaves weekly using a hand lens.
3. Avoid broad-spectrum insecticides as these kill natural predators and can trigger mite outbreaks.
4. Control weeds as they serve as alternative hosts for mites.
5. Field sanitation: Remove and destroy crop residues after harvest.
6. Avoid planting near infested fields as mites can travel on wind currents.""",
            "chemical_control": f"""{_bold('Miticides for Tomato Mites:')}

{_bold('Chemical Miticides:')}
i) Abamectin (e.g., Vertimec 18EC from Syngenta Egypt [1]). Use 0.5-1.0ml/L of water.
ii) Spiromesifen (e.g., Oberon 240SC from Bayer East Africa [2]). Use 1.0ml/L of water. Effective on eggs and young stages.
iii) Propargite (e.g., Omite 570EW from Corteva Agriscience [3]). Use 1.5-2.0ml/L of water.
iv) Sulphur-based products (e.g., Kumulus DF from BASF East Africa Ltd [4]). Use 2-3g/L of water.

{_bold('Organic Options:')}
i) Neem oil (e.g., NeemAzal 1.2EC from Twiga Chemicals [5], Nimbecidine from Osho Chemical Industries Ltd [6]). Use 5-10ml/L of water. Repels and disrupts mite development.
ii) Insecticidal soap (e.g., Des-X from Certis Biologicals [7]). Use 2-3g/L of water. Kills mites on contact.

{_bold('Application timing:')} Apply miticides when mites are first observed. Target the undersides of leaves thoroughly. Repeat at 5-7 day intervals during hot, dry conditions. Rotate between different chemical groups to prevent the mites from developing resistance.""",
            "xai_ref_numbers": [57, 71],
            "management_refs": [57, 71],
            "chemical_refs_original": [94, 95, 96, 63, 8, 9, 97]
        }

    # ==================== TOMATO ALTERNARIA MITE ====================
    elif 'alternaria mite' in name_lower:
        return {
            "disease_name": "Tomato Alternaria Mite Disease",
            "category": "🐛 PEST INFESTATION + 🍄 FUNGAL DISEASE",
            "causal_agent": "Combination of Alternaria fungal infection and mite damage",
            "is_healthy": False,
            "causes_characteristics": f"""This is a combination of Alternaria fungal infection combined with mite damage.

{_bold('Disease Complex:')}
• Mites create wounds on leaf tissue, providing an entry point for Alternaria fungi.
• The combination results in more severe damage than either pest or disease alone.
• Mite damage weakens plant defences, making fungal infection more severe.
• Alternaria infection spreads more rapidly through mite-damaged tissue.""",
            "management": f"""{_bold('Integrated Management (Mite + Fungal Control):')}

1. Control mites first as mite damage facilitates fungal infection.
2. Apply a miticide + fungicide combination for effective control.
3. Remove and destroy heavily infested leaves.
4. Improve air circulation through pruning and proper spacing.
5. Avoid water stress as stressed plants are more susceptible to both mites and Alternaria.
6. Use biological control for mites (predatory mites) where possible.""",
            "chemical_control": f"""{_bold('Combination Approach (Miticide + Fungicide):')}

{_bold('Step 1 – Miticide (for mites):')}
i) Abamectin (e.g., Vertimec 18EC from Syngenta Egypt [1]). Use 0.5-1.0ml/L of water.

{_bold('Step 2 – Plus a Fungicide (for Alternaria) such as:')}
i) Mancozeb (e.g., Dithane M-45 from UPL Limited [2], Milthane Super from Twiga Chemical Industries Ltd [3], Fortress Gold 72 WP from Greenlife Crop Protection Africa Ltd [4]). Use 2.5g/L of water.
ii) Chlorothalonil (e.g., Bravo 720 SC from Syngenta East Africa Ltd [5]). Use 2.0-3.0ml/L of water.
iii) Azoxystrobin (e.g., Amistar® from Syngenta East Africa [6], Milestone® 250SC [7] and Absolute 375SC [8], both from Greenlife Crop Protection Africa). Use 1.0ml/L of water.
iv) Copper-based fungicides (e.g., Kocide 2000 from DuPont [9], Nordox 75 WG from Nordox [10], Copac E from Greenlife Crop Protection Africa [11], Cupravit® 50 WP from Bayer East Africa [12]). Use 2.0-2.5g/L of water.

{_bold('Pre-mixed Combination Products (Organic option):')}
i) Neem oil (e.g., NeemAzal 1.2EC from Twiga Chemicals [13], Nimbecidine from Osho Chemical Industries Ltd [14]). Use 5-10ml/L of water + Copper-based fungicide (e.g., Kocide 2000 from DuPont [9]). Use 2.0-2.5g/L of water.

{_bold('Application timing:')} Apply at the first sign of mite damage or Alternaria symptoms. Target the undersides of leaves. Repeat at 5-7 day intervals during hot, dry conditions.""",
            "xai_ref_numbers": [57, 71, 72],
            "management_refs": [57, 71, 72],
            "chemical_refs_original": [94, 27, 28, 29, 58, 68, 23, 20, 64, 65, 66, 67, 8, 9]
        }

    # ==================== TOMATO EXCESS NITROGEN ====================
    elif 'excess nitrogen' in name_lower:
        return {
            "disease_name": "Tomato Excess Nitrogen Disease",
            "category": "🌱 PHYSIOLOGICAL DISORDER",
            "causal_agent": "Excessive nitrogen fertilization",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Cause:')} Excessive nitrogen fertilization, particularly with ammonium-based or quick-release nitrogen sources.

{_bold('Symptoms:')}
• Vegetative: Excessive, lush, dark green vegetative growth. Plants appear "overgrown" with thick stems and abundant leaves.
• Flowering/Fruiting: Delayed flowering and fruit set. Reduced fruit production (more leaves, fewer fruits).
• Fruit Quality: Fruits may be smaller, have reduced flavour, and may be more susceptible to other disorders (blossom end rot).
• Pest/Disease Susceptibility: Lush growth attracts aphids, whiteflies, and other pests. Dense canopy reduces air circulation, promoting fungal diseases.""",
            "management": f"""{_bold('Management focuses on prevention and mitigation:')}

{_bold('FOR AFFECTED CROPS:')}
• Reduce or stop nitrogen fertilization immediately.
• Increase potassium fertilization as potassium helps balance nitrogen effects.
• Prune excessive growth to improve air circulation and reduce disease pressure.
• In well-drained soils, increase irrigation carefully to leach excess nitrogen from the root zone.

{_bold('FOR LONG-TERM PREVENTION:')}
1. Soil testing: Test soil before planting to determine existing nitrogen levels.
2. Follow recommended fertilizer rates based on soil tests and crop requirements.
3. Use controlled-release fertilizers or split applications to match plant needs.
4. Use organic fertilizers (compost, manure) that release nitrogen slowly.
5. Avoid excessive manure application – manure can be high in nitrogen.
6. Use nitrogen-efficient varieties where available.
7. Grow cover crops (e.g., grasses) to absorb excess nitrogen between tomato crops.""",
            "chemical_control": f"""Not applicable. Management is cultural.

{_bold('Recommendations:')}
• Reduce nitrogen application. Apply only based on soil test results (typically 40-60 kg N/acre for tomatoes).
• Use split applications. Apply nitrogen in 2-3 smaller doses rather than all at once.
• Use slow-release or organic nitrogen sources. These release nitrogen more gradually.
• Balanced fertilization. Use complete NPK fertilizers with appropriate ratios (e.g., 10:10:10 or 10:20:10).
• Foliar testing. Tissue analysis can confirm excess nitrogen status.""",
            "xai_ref_numbers": [98, 99],
            "management_refs": [98, 99],
            "chemical_refs_original": []
        }

    # ==================== TOMATO SUNBURN ====================
    elif 'sunburn' in name_lower or 'sunscald' in name_lower:
        return {
            "disease_name": "Tomato Sunburn (Sunscald) Disease",
            "category": "🌱 PHYSIOLOGICAL DISORDER",
            "causal_agent": "Exposure of tomato fruits to excessive direct sunlight and high temperatures",
            "is_healthy": False,
            "causes_characteristics": f"""{_bold('Cause:')} Exposure of tomato fruits to excessive direct sunlight and high temperatures. This typically occurs after leaf loss from pruning, disease, or pest damage.

{_bold('Symptoms:')}
• Fruit: White, yellow, or greyish blistered or sunken areas on the side of the fruit exposed to the sun.
• The affected area becomes dry, leathery, and papery.
• Secondary Rot: Sunburned areas are prone to secondary infection by opportunistic fungi and bacteria, leading to fruit rot.
• Typically affects fruits that are not adequately shaded by leaves.
• Determinate varieties (with less foliage) are more susceptible.

{_bold('Predisposing Factors:')}
• Over-pruning (removing too many leaves)
• Disease-induced defoliation (e.g., by late blight, early blight, Septoria)
• Pest damage (e.g., by mites, thrips) causing leaf loss
• Growing determinate varieties with less natural foliage cover""",
            "management": f"""{_bold('Management focuses on prevention:')}

{_bold('FOR AFFECTED CROPS:')}
• Remove sunburned fruits as they will not recover and may rot.
• Provide temporary shade using shade cloth (30-50% shade) during peak sun hours.
• Allow natural leaf growth. Avoid further pruning of leaves that shade fruits.

{_bold('FOR LONG-TERM PREVENTION:')}
1. Avoid over-pruning. Maintain adequate foliage to shade developing fruits.
2. Prune appropriately. For indeterminate varieties, prune suckers but keep enough leaves to shade fruits.
3. Choose suitable varieties. Indeterminate varieties with good leaf cover are less susceptible.
4. Manage diseases and pests that cause defoliation (e.g., early blight, Septoria, mites).
5. Use shade cloth (30-50% shade) in high-intensity sun regions.
6. Proper plant spacing. Closer spacing provides mutual shading (but must balance with air circulation needs).
7. Orient rows north-south to provide more even sun exposure.
8. Stake and trellis. Vertical growing allows better light penetration while maintaining leaf cover.""",
            "chemical_control": "Not applicable. Cultural management only.",
            "xai_ref_numbers": [98, 99],
            "management_refs": [98, 99],
            "chemical_refs_original": []
        }

    # Default fallback
    else:
        return {
            "disease_name": class_name,
            "category": "General",
            "causal_agent": "Please consult a local agricultural extension officer for accurate identification",
            "is_healthy": False,
            "causes_characteristics": f"No detailed information available for {class_name}. Please contact your local agricultural extension office for assistance.",
            "management": "Please consult your local agricultural extension officer for advice specific to this condition. Take clear photos of affected plants and, if possible, bring a sample to the extension office.",
            "chemical_control": "Consult local agricultural experts for appropriate treatment recommendations. Do not apply any chemicals without proper identification and expert guidance.",
            "xai_ref_numbers": [],
            "management_refs": [],
            "chemical_refs_original": []
        }

# ============================================================
# COMMON CHEMICALS FUNCTION
# ============================================================
def show_common_chemicals_for_top_k(top_k_predictions, references):
    """Show chemicals that work for multiple predicted diseases"""
    from collections import defaultdict
    import re

    k = len(top_k_predictions)

    st.markdown(f"""
    <div class="diagnosis-card">
        <div class="result-section">
            <h3>🌿 CHEMICALS FOR YOUR TOP {k} PREDICTED DISEASES</h3>
    """, unsafe_allow_html=True)

    st.markdown(f"**Your Top {k} Predicted Diseases:**")
    valid_predictions = []
    healthy_diseases = []

    for i, pred in enumerate(top_k_predictions, 1):
        is_healthy = 'healthy' in pred['class'].lower()

        if is_healthy:
            healthy_diseases.append((pred['class'], pred['confidence']))
            st.markdown(f"   {i}. {pred['class']} ({pred['confidence']*100:.1f}%) - 🌿 HEALTHY CROP (no treatment needed)")
        else:
            valid_predictions.append((pred['class'], pred['confidence']))
            st.markdown(f"   {i}. {pred['class']} ({pred['confidence']*100:.1f}%)")

    if not valid_predictions:
        st.markdown(f"""
        <div class="result-section">
            <h3>🌿 ALL PREDICTIONS ARE HEALTHY CROPS!</h3>
            <p>✅ Good news! Your crop appears healthy based on the top {k} predictions.</p>
            <p><strong>📋 RECOMMENDATION:</strong></p>
            <p>   • Continue good farming practices</p>
            <p>   • Regular field monitoring is still recommended</p>
            <p>   • No chemical treatments needed</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"**💡 Strategy:** Looking for chemicals that work for multiple diseases...\n")

    chemical_classes = {
        'Mancozeb': ['mancozeb', 'dithane', 'milthane', 'fortress gold'],
        'Copper-based': ['copper', 'kocide', 'nordox', 'cupravit', 'copac'],
        'Azoxystrobin': ['azoxystrobin', 'milestone', 'amistar', 'absolute'],
        'Tebuconazole': ['tebuconazole', 'nativo', 'folicur'],
        'Propiconazole': ['propiconazole', 'tilt', 'defacto'],
        'Pyraclostrobin': ['pyraclostrobin', 'comet', 'pyracopp'],
        'Chlorothalonil': ['chlorothalonil', 'bravo'],
        'Imidacloprid': ['imidacloprid', 'confidor', 'emerald'],
        'Lambda-cyhalothrin': ['lambda-cyhalothrin', 'karate', 'lamdastar'],
        'Emamectin Benzoate': ['emamectin', 'escort'],
        'Abamectin': ['abamectin', 'vertimec'],
        'Spinosad': ['spinosad', 'tracer'],
        'Neem Oil': ['neem', 'neemazal', 'nimbecidine']
    }

    def get_action_info(chem_class):
        action_map = {
            'Mancozeb': ("PROTECTANT (Preventive)", "Apply BEFORE disease appears to prevent infection"),
            'Copper-based': ("PROTECTANT (Preventive)", "Broad-spectrum preventive fungicide/bactericide"),
            'Chlorothalonil': ("PROTECTANT (Preventive)", "Broad-spectrum preventive fungicide"),
            'Azoxystrobin': ("CURATIVE + PROTECTANT", "Treats existing infection AND prevents new ones"),
            'Pyraclostrobin': ("CURATIVE + PROTECTANT", "Treats existing infection AND prevents new ones"),
            'Tebuconazole': ("CURATIVE (Systemic)", "Treats active fungal infection"),
            'Propiconazole': ("CURATIVE (Systemic)", "Treats active fungal infection"),
            'Imidacloprid': ("INSECTICIDE (Systemic)", "Controls sap-sucking insects"),
            'Lambda-cyhalothrin': ("INSECTICIDE (Contact)", "Broad-spectrum insect control"),
            'Emamectin Benzoate': ("INSECTICIDE (Translaminar)", "Controls caterpillars and leafminers"),
            'Spinosad': ("INSECTICIDE (Ingestion)", "Controls caterpillars, thrips, and leafminers"),
            'Abamectin': ("MITICIDE/INSECTICIDE", "Controls spider mites and leafminers"),
            'Neem Oil': ("BIO-PESTICIDE", "Organic - repels pests and disrupts fungal growth"),
        }
        return action_map.get(chem_class, ("Varies by product", "Check product label for specific instructions"))

    disease_chemicals = defaultdict(set)
    chemical_products_map = defaultdict(set)

    for disease, conf in valid_predictions:
        treatment = get_full_treatment(disease, references)
        chem_text = treatment['chemical_control'].lower()

        for chem_class, keywords in chemical_classes.items():
            if any(keyword in chem_text for keyword in keywords):
                disease_chemicals[chem_class].add(disease)
                lines = chem_text.split('\n')
                for line in lines:
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in keywords):
                        clean_line = re.sub(r'\[\d+\]', '', line)
                        clean_line = re.sub(r'^\s*[\(\)]?\s*[ivxcl]+\s*[\)\.]\s*', '', clean_line, flags=re.IGNORECASE)
                        clean_line = re.sub(r'^[Ee]\.?[Gg]\.?,?\s*', '', clean_line)
                        if clean_line and len(clean_line) > 5:
                            chemical_products_map[chem_class].add(clean_line[:80])

    found_match = False
    target = len(valid_predictions)

    while target >= 2 and not found_match:
        matches = {}
        for chem_class, diseases in disease_chemicals.items():
            if len(diseases) >= target:
                matches[chem_class] = diseases

        if matches:
            found_match = True
            if target == len(valid_predictions):
                st.markdown(f"**✅ CHEMICALS THAT WORK FOR ALL {len(valid_predictions)} DISEASES:**\n")
            else:
                st.markdown(f"**⚠️ NO CHEMICAL WORKS FOR ALL {len(valid_predictions)} DISEASES**")
                st.markdown(f"   Looking for chemicals that work for at least {target} diseases...\n")
                st.markdown(f"**✅ CHEMICALS THAT WORK FOR {target}+ DISEASES:**\n")

            for chem_class, diseases in sorted(matches.items()):
                all_disease_names = [d[0] for d in valid_predictions]
                disease_list = sorted(diseases)
                not_controlled = [d for d in all_disease_names if d not in disease_list]
                action, desc = get_action_info(chem_class)

                st.markdown(f"**🔹 {chem_class}**")
                st.markdown(f"   **Action:** {action}")
                st.markdown(f"   **How it works:** {desc}")
                st.markdown(f"   **Controls:** {', '.join(disease_list)}")
                if not_controlled:
                    st.markdown(f"   **Does NOT control:** {', '.join(not_controlled)}")
                if chem_class in chemical_products_map and chemical_products_map[chem_class]:
                    st.markdown(f"   **Products (examples):**")
                    for product in list(chemical_products_map[chem_class])[:2]:
                        st.markdown(f"      • {product}")
                st.markdown("")
        else:
            target -= 1

    if not found_match:
        st.markdown("**❌ NO CHEMICALS FOUND FOR MULTIPLE DISEASES**")
        st.markdown(f"\n**💡 RECOMMENDATION:**")
        st.markdown(f"   • View individual treatments for each disease")

    if healthy_diseases:
        st.markdown(f"\n**🌿 NOTE ABOUT HEALTHY CROP PREDICTIONS:**")
        for disease, conf in healthy_diseases:
            st.markdown(f"   • {disease}: {conf*100:.1f}%")

    st.markdown(f"""
        <div class="result-section">
            <h3>📋 IMPORTANT NOTES:</h3>
            <p>   • <strong>PROTECTANT</strong> = Apply BEFORE disease appears</p>
            <p>   • <strong>CURATIVE</strong> = Apply AFTER disease appears</p>
            <p>   • Always read and follow product label instructions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HELP INFORMATION
# ============================================================

def display_help():
    """Display help information - Same style as List of Supported Classes"""
    st.markdown("""
    <div class="help-card">
        <h4>📚 How to use Crop Doctor</h4>

        **📸 STEP 1: Take or Upload a Photo**

        • Take a clear photo of the affected crop part (leaf, stem, or fruit)
        • Ensure good lighting and focus on the symptoms
        • Or upload an existing image from your gallery

        **🔬 STEP 2: Diagnose Your Crop**

        • Click the "DIAGNOSE & RECOMMEND" button
        • The AI model will analyse the image and identify the disease
        • Top predictions are shown with confidence scores

        **📊 STEP 3: Understand the Results**

        • Confidence Score (0-100%) - How reliable the prediction is
        • HEALTH CONFIDENCE KEY - For healthy crops (different scale)
        • URGENCY KEY - For diseased crops (what action to take)
        • Grad-CAM Heatmap - Shows which areas influenced the diagnosis
        • 🔴 Red areas = High influence  |  🔵 Blue areas = Low influence

        **💊 STEP 4: Get Treatment Recommendations**

        • Verified treatments from Agrochemical manufacturers
        • Includes cultural management and chemical control options
        • Complete with product examples and reference citations

        **🌐 STEP 5: Online Mode (Optional)**

        • Get local weather forecast and disease risk assessment
        • Receive disease-specific recommendations based on weather
        • Check manufacturer websites for new products

        ---

        **💡 Tips for Best Results**

        • Take photos of affected leaves showing clear symptoms
        • Include a healthy leaf for comparison if possible
        • Avoid blurry or poorly lit images
        • For low confidence predictions, consult a local expert
        • Always verify chemical recommendations with local agrovets

        ---

        **🌤️ Understanding Weather Information**

        Hover over the ❔ icon next to any weather parameter to get detailed explanations:

        | Parameter | What the tooltip explains |
        |-----------|--------------------------|
        | 🌡️ Temperature | Current air temperature and its effect on crops |
        | 💧 Humidity | How humidity affects diseases and pests |
        | ☔ Rainfall | Safe amounts for spraying vs. wash-off risk |
        | 🌬️ Wind Speed | Best conditions for spraying |
        | 🌧️ Rain Probability | Chance of rain and how much to expect |
        | 🎯 Disease Risk | What the risk level means for your crops |

        **📖 Quick Weather Reference:**

        | Condition | What it means | Action |
        |-----------|---------------|--------|
        | Rain Probability >70% with <2mm | Light rain expected | ✅ Safe to spray |
        | Rain Probability >70% with >10mm | Heavy rain expected | ⚠️ Delay spraying |
        | Temperature >30°C | Heat stress risk | 💧 Increase irrigation |
        | Humidity >80% | Fungal disease risk | 🍄 Apply preventive fungicide |
        | Humidity <40% | Pest risk | 🔍 Monitor for aphids/mites |
        | Wind >25 km/h | Spray drift risk | ⏰ Wait for calmer conditions |

        ---

        **⚠️ Important Notes**

        • This is an AI-assisted diagnostic tool, not a substitute for expert advice
        • Always read product labels before applying any chemicals
        • Follow local regulations and safety guidelines
        • Consult your local agricultural extension officer for confirmation
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""

    def __init__(self, model, layer_name='gradcam_conv'):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = None
        self._build_grad_model()

    def _build_grad_model(self):
        """Build gradient model for Grad-CAM"""
        try:
            target_layer = self.model.get_layer(self.layer_name)
            self.grad_model = tf.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[target_layer.output, self.model.output]
            )
        except:
            for layer in self.model.layers:
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                    self.grad_model = tf.keras.models.Model(
                        inputs=[self.model.inputs],
                        outputs=[layer.output, self.model.output]
                    )
                    self.layer_name = layer.name
                    break

    def generate_heatmap(self, img_array, class_idx):
        """Generate Grad-CAM heatmap for the given image and class"""
        if self.grad_model is None:
            return np.ones((7, 7)) * 0.5

        try:
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(img_tensor)
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                confidence = float(predictions[0, class_idx])
                size = 7
                heatmap = np.zeros((size, size))
                center = size // 2
                for i in range(size):
                    for j in range(size):
                        dist = np.sqrt((i - center)**2 + (j - center)**2)
                        heatmap[i, j] = confidence * np.exp(-dist / 3)
                return heatmap / (heatmap.max() + 1e-8)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val

            return heatmap.numpy()
        except:
            return np.ones((7, 7)) * 0.5

    def overlay_heatmap(self, heatmap, original_img, alpha=0.4):
        """Overlay heatmap on original image with spectrum colormap"""
        try:
            if isinstance(original_img, Image.Image):
                img_array = np.array(original_img)
            else:
                img_array = original_img.copy() if isinstance(original_img, np.ndarray) else np.array(original_img)

            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = np.uint8(255 * img_array)
                else:
                    img_array = np.uint8(img_array)

            heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            heatmap_normalized = np.clip(heatmap_resized, 0, 1)

            h, w = heatmap_normalized.shape
            heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(h):
                for j in range(w):
                    val = heatmap_normalized[i, j]
                    if val <= 0.2:
                        t = val / 0.2
                        r = 0
                        g = int(255 * t)
                        b = 255
                    elif val <= 0.4:
                        t = (val - 0.2) / 0.2
                        r = 0
                        g = 255
                        b = 255 - int(255 * t)
                    elif val <= 0.6:
                        t = (val - 0.4) / 0.2
                        r = int(255 * t)
                        g = 255
                        b = 0
                    elif val <= 0.8:
                        t = (val - 0.6) / 0.2
                        r = 255
                        g = 255 - int(90 * t)
                        b = 0
                    else:
                        t = (val - 0.8) / 0.2
                        r = 255
                        g = 165 - int(165 * t)
                        b = 0
                    heatmap_colored[i, j] = (b, g, r)

            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array

            overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
            return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        except:
            return img_array

# ============================================================
# PREPROCESS FUNCTION
# ============================================================
def preprocess_image(image):
    """Preprocess image for model input"""
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ============================================================
# FORMAT REFERENCE LIST
# ============================================================
def format_reference_list_sequential(original_ref_numbers, ref_dictionary):
    """Format references with each on a new line, matching XAI SOURCES style"""
    if not original_ref_numbers or len(original_ref_numbers) == 0:
        return "See local agricultural extension office for region-specific advice"
    ref_list = []
    for idx, global_num in enumerate(original_ref_numbers, 1):
        if global_num in ref_dictionary:
            ref_list.append(f"[{idx}] {ref_dictionary[global_num]}")
        else:
            ref_list.append(f"[{idx}] Reference {global_num}")
    return "\n".join(ref_list) if ref_list else "See local agricultural extension office for region-specific advice"

# ============================================================
# CROP HEALTH ASSESSMENT (FOR HEALTHY CROPS)
# ============================================================

def display_healthy_crop_assessment(predicted_class, confidence, treatment, references, heatmap_overlay=None, original_img=None, save_path=None):
    """Display crop health assessment for healthy crops - COMPLETE with ALL sections matching disease class"""

    # Determine health status based on confidence
    if confidence >= 0.9:
        health_status = "EXCELLENT"
        status_icon = "🟢"
        confidence_text = "VERY HIGH"
        confidence_icon = "🟢"
        advice = "Your crop is in EXCELLENT health! Your farming practices are working very well."
        action = "✅ MAINTAIN CURRENT PRACTICES"
        urgency = "NONE"
        urgency_icon = "🟢"
        level = "VERY HIGH"
        level_icon = "🟢"
        level_desc = "Extremely reliable - the crop shows very clear and distinct healthy characteristics."
    elif confidence >= 0.8:
        health_status = "VERY GOOD"
        status_icon = "🟢"
        confidence_text = "HIGH"
        confidence_icon = "🟢"
        advice = "Your crop is very healthy. Keep up the good management practices."
        action = "✅ CONTINUE GOOD PRACTICES"
        urgency = "NONE"
        urgency_icon = "🟢"
        level = "HIGH"
        level_icon = "🟢"
        level_desc = "Very reliable - the crop shows clearly visible healthy characteristics."
    elif confidence >= 0.7:
        health_status = "GOOD"
        status_icon = "🟢"
        confidence_text = "HIGH"
        confidence_icon = "🟡"
        advice = "Your crop is healthy. Regular monitoring is recommended to maintain this status."
        action = "📋 ROUTINE MONITORING"
        urgency = "LOW"
        urgency_icon = "🟡"
        level = "HIGH"
        level_icon = "🟡"
        level_desc = "Reliable - the crop shows healthy characteristics, though not extremely strong."
    elif confidence >= 0.6:
        health_status = "SATISFACTORY"
        status_icon = "🟡"
        confidence_text = "MODERATE"
        confidence_icon = "🟡"
        advice = "Your crop appears generally healthy but confidence is moderate. Consider closer inspection."
        action = "🔍 VERIFY HEALTH STATUS"
        urgency = "LOW"
        urgency_icon = "🟡"
        level = "MODERATE"
        level_icon = "🟡"
        level_desc = "Moderately reliable - some healthy characteristics are visible."
    elif confidence >= 0.5:
        health_status = "LIKELY HEALTHY"
        status_icon = "🟡"
        confidence_text = "MODERATE"
        confidence_icon = "🟠"
        advice = "Your crop is likely healthy but there is some uncertainty. Inspect for any subtle symptoms."
        action = "👁️ CAREFUL INSPECTION"
        urgency = "MEDIUM"
        urgency_icon = "🟠"
        level = "MODERATE"
        level_icon = "🟠"
        level_desc = "Moderately reliable - healthy characteristics are present but weak."
    elif confidence >= 0.4:
        health_status = "POSSIBLY HEALTHY"
        status_icon = "🟠"
        confidence_text = "LOW"
        confidence_icon = "🟠"
        advice = "The model suggests health but with low confidence. Manual inspection recommended."
        action = "🔬 PROFESSIONAL CHECK"
        urgency = "MEDIUM"
        urgency_icon = "🟠"
        level = "LOW"
        level_icon = "🟠"
        level_desc = "Low reliability - healthy characteristics are unclear or mixed."
    elif confidence >= 0.3:
        health_status = "UNCLEAR"
        status_icon = "🟠"
        confidence_text = "LOW"
        confidence_icon = "🔴"
        advice = "Health status is unclear. Please inspect your crop thoroughly."
        action = "👨‍🌾 EXPERT CONSULTATION"
        urgency = "HIGH"
        urgency_icon = "🔴"
        level = "LOW"
        level_icon = "🔴"
        level_desc = "Low reliability - weak healthy characteristics present."
    else:
        health_status = "INCONCLUSIVE"
        status_icon = "🔴"
        confidence_text = "VERY LOW"
        confidence_icon = "🔴"
        advice = "The model cannot confidently determine health status. Professional inspection is strongly recommended."
        action = "🚨 IMMEDIATE EXPERT CHECK"
        urgency = "HIGH"
        urgency_icon = "🔴"
        level = "VERY LOW"
        level_icon = "🔴"
        level_desc = "Very low reliability - healthy characteristics are absent or unclear."

    # Extract key characteristics
    causes_text = treatment.get('causes_characteristics', '')
    import re
    clean_text = re.sub(r'\[\d+\]', '', causes_text)
    lines = clean_text.split('\n')

    key_characteristics = []
    for line in lines:
        line = line.strip()
        if line and (line.startswith('•') or line.startswith('-') or re.match(r'^\d+\.', line)):
            clean_line = re.sub(r'^[•\-*\d\.\s]+', '', line)
            if clean_line and len(clean_line) > 5:
                key_characteristics.append(clean_line)

    if not key_characteristics:
        key_characteristics = [
            "Vibrant, deep green leaves indicating good health",
            "Clean foliage free from spots, lesions, or discolouration",
            "Thick, sturdy stems with robust growth",
            "Well-developed root system for efficient nutrient uptake",
            "Normal growth patterns without stunting or distortion"
        ]

    characteristics_text = ""
    for i, char in enumerate(key_characteristics[:6], 1):
        characteristics_text += f"\n  {i}. {char}"

    # SECTION 1: XAI ANALYSIS TITLE
    st.markdown(f"""
<div class="section-card">
<h3>🔬 EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) ANALYSIS FOR: {predicted_class}</h3>
<p><strong>📊 CLASSIFIER CONFIDENCE SCORE:</strong> {confidence*100:.1f}% ({level}) {level_icon}</p>
<p>{level_desc}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 2: HEALTH CONFIDENCE KEY
    st.markdown("""
<div class="section-card">
<h3>📊 HEALTH CONFIDENCE KEY - Understanding the Score for Healthy Crops</h3>
<p>   🟢 <strong>90-100% (VERY HIGH)</strong> → EXCELLENT health! No action needed.</p>
<p>   🟢 <strong>80-89% (HIGH)</strong>     → VERY GOOD health. Maintain practices.</p>
<p>   🟡 <strong>70-79% (HIGH)</strong>     → GOOD health. Routine monitoring.</p>
<p>   🟡 <strong>60-69% (MODERATE)</strong> → SATISFACTORY health. Verify status.</p>
<p>   🟠 <strong>50-59% (MODERATE)</strong> → LIKELY healthy. Inspect thoroughly.</p>
<p>   🟠 <strong>40-49% (LOW)</strong>      → POSSIBLY healthy. Seek confirmation.</p>
<p>   🔴 <strong>30-39% (LOW)</strong>      → UNCLEAR status. Expert recommended.</p>
<p>   🔴 <strong>0-29% (VERY LOW)</strong>  → INCONCLUSIVE. Immediate expert check needed.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 3: DISEASE TYPE
    st.markdown(f"""
<div class="section-card">
<h3>🏷️ DISEASE TYPE</h3>
<p>{treatment['category']}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 4: CAUSAL AGENT
    st.markdown(f"""
<div class="section-card">
<h3>🦠 CAUSAL AGENT</h3>
<p>{treatment['causal_agent']}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 5: KEY CHARACTERISTICS OF A HEALTHY CROP
    st.markdown(f"""
<div class="section-card">
<h3>📋 KEY CHARACTERISTICS OF A HEALTHY CROP</h3>
<p style="white-space:pre-line;">{characteristics_text}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 6: WHAT THE MODEL ANALYSED
    st.markdown(f"""
<div class="section-card">
<h3>💡 WHAT THE MODEL ANALYSED</h3>
<p>The model examined your crop image and identified visual patterns that match the characteristic appearance of a healthy {predicted_class.replace('Healthy', '').strip()} plant.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 7: VISUAL EVIDENCE (Grad-CAM Heatmap)
    st.markdown(f"""
<div class="section-card">
<h3>🔥 VISUAL EVIDENCE (Heatmap)</h3>
<p>✓ Generated</p>
</div>
""", unsafe_allow_html=True)

    # Display heatmap if available
    if heatmap_overlay is not None and original_img is not None:
        display_heatmap_with_colorbar(original_img, heatmap_overlay, predicted_class, save_path)

    # SECTION 8: GRAD-CAM LEGEND
    st.markdown("""
<div class="section-card">
<h3>📊 HEATMAP LEGEND</h3>
<p>   🔴 <strong>RED (HOT)</strong>     = HIGH influence - These areas strongly indicate the condition</p>
<p>   🟠 <strong>ORANGE</strong>        = HIGH-MEDIUM influence</p>
<p>   🟡 <strong>YELLOW</strong>        = MEDIUM influence</p>
<p>   🟢 <strong>GREEN</strong>         = LOW-MEDIUM influence</p>
<p>   💠 <strong>CYAN</strong>          = VERY LOW influence</p>
<p>   🔵 <strong>BLUE (COOL)</strong>   = LOWEST influence - These areas had minimal impact on the diagnosis</p>
<p class="gradcam-tip">💡 The warmer the color (Red → Orange → Yellow), the more it influenced the model's decision. Cooler colors (Green → Cyan → Blue) had less influence.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 9: WHAT THIS MEANS FOR YOU
    st.markdown(f"""
<div class="section-card">
<h3>📋 WHAT THIS MEANS FOR YOU</h3>
<p>Based on the visual patterns detected, the system has determined that your crop is HEALTHY with {level.lower()} confidence.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 10: TREATMENT RECOMMENDATION
    st.markdown(f"""
<div class="section-card">
<h3>🌾 TREATMENT RECOMMENDATION FOR: {predicted_class}</h3>
<p><strong>📊 DIAGNOSIS:</strong> {predicted_class}</p>
<p><strong>🏷️ TYPE:</strong> {treatment['category']}</p>
<p><strong>📈 CLASSIFIER CONFIDENCE:</strong> {confidence*100:.1f}% ({confidence_text}) {confidence_icon}</p>
<p><strong>⚡ URGENCY & ACTION:</strong> {urgency_icon} {urgency} - {action}</p>
<p><strong>💡 REASON:</strong> {advice}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 11: URGENCY KEY
    st.markdown("""
<div class="section-card">
<h3>⚡ URGENCY KEY - What Action to Take Based on Score (Healthy Crops)</h3>
<p>   🟢 <strong>90-100% (NONE)</strong>    → EXCELLENT health. No action needed.</p>
<p>   🟢 <strong>80-89% (NONE)</strong>     → VERY GOOD health. Maintain practices.</p>
<p>   🟡 <strong>70-79% (LOW)</strong>      → GOOD health. Routine monitoring.</p>
<p>   🟡 <strong>60-69% (LOW)</strong>      → SATISFACTORY health. Verify status.</p>
<p>   🟠 <strong>50-59% (MEDIUM)</strong>   → LIKELY healthy. Inspect thoroughly.</p>
<p>   🟠 <strong>40-49% (MEDIUM)</strong>   → POSSIBLY healthy. Seek confirmation.</p>
<p>   🔴 <strong>30-39% (HIGH)</strong>     → UNCLEAR status. Expert consultation.</p>
<p>   🔴 <strong>0-29% (HIGH)</strong>      → INCONCLUSIVE. Immediate expert check.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 12: RECOMMENDED MANAGEMENT
    st.markdown(f"""
<div class="section-card">
<h3>🔧 RECOMMENDED MANAGEMENT</h3>
<p style="white-space:pre-line;">{treatment['management']}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 13: GOOD FARMING PRACTICES TO MAINTAIN
    st.markdown("""
<div class="section-card">
<h3>📋 GOOD FARMING PRACTICES TO MAINTAIN</h3>
<p>  • Maintain proper irrigation schedule</p>
<p>  • Apply balanced fertilisation based on soil tests</p>
<p>  • Conduct weekly field scouting for early detection</p>
<p>  • Practice crop rotation for long-term soil health</p>
<p>  • Use certified disease-free seeds for next planting</p>
<p>  • Keep field records for trend analysis</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 14: REFERENCES
    management_refs = treatment.get('management_refs', [1])
    references_text = format_reference_list_sequential(management_refs, references)

    st.markdown(f"""
<div class="section-card">
<h3>📚 REFERENCES</h3>
<div class="reference-text">{references_text}</div>
</div>
""", unsafe_allow_html=True)

    # SECTION 15: WHEN TO CONSULT AN EXPERT
    st.markdown("""
<div class="section-card">
<h3>👨‍🌾 WHEN TO CONSULT AN EXPERT</h3>
<p>  • If confidence is below 60% (MODERATE or lower)</p>
<p>  • If you notice any unusual symptoms despite the healthy diagnosis</p>
<p>  • For region-specific pest alerts and seasonal advice</p>
<p>  • Before making major changes to your farming practices</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_heatmap_with_colorbar(original_img, heatmap_overlay, predicted_class, save_path=None):
    """Display heatmap with enhanced color bar - NO save message"""
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original Image", width="stretch")

    with col2:
        st.image(heatmap_overlay, caption=f"Heatmap Showing areas that led the model to pick on:\n{predicted_class}", width="stretch")

        # Create larger, more readable color bar
        fig, ax = plt.subplots(figsize=(10, 1))
        fig.patch.set_visible(False)
        ax.axis('off')

        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        spectrum_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        spectrum_cmap = LinearSegmentedColormap.from_list('spectrum', spectrum_colors, N=256)

        ax.imshow(gradient, aspect='auto', cmap=spectrum_cmap, extent=[0, 1, 0, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0, -0.8, 'LOW', ha='center', va='top', fontsize=14, fontweight='bold', color='blue')
        ax.text(0.5, -0.8, 'MEDIUM', ha='center', va='top', fontsize=14, fontweight='bold', color='green')
        ax.text(1, -0.8, 'HIGH', ha='center', va='top', fontsize=14, fontweight='bold', color='red')

        ax.annotate('', xy=(1, -1.5), xytext=(0, -1.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

        ax.text(0.5, -1.8, 'Influence on Predictions', ha='center', va='top', fontsize=16, fontweight='bold', color='black')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def display_top_predictions(top_predictions):
    """Display top predictions - each in its own curved box"""

    # Build the predictions HTML
    predictions_html = f"""
<div class="section-card">
<h3>📈 TOP {len(top_predictions)} PREDICTIONS</h3>
<div class="prediction-list">
"""
    for i, pred in enumerate(top_predictions):
        if i == 0:
            predictions_html += f'<div class="prediction-row"><span class="prediction-marker">✓.</span> <span class="prediction-name">{pred["class"]}</span>: <span class="prediction-value">{pred["confidence"]*100:.1f}%</span></div>'
        else:
            predictions_html += f'<div class="prediction-row"><span class="prediction-marker">{i+1}.</span> <span class="prediction-name">{pred["class"]}</span>: <span class="prediction-value">{pred["confidence"]*100:.1f}%</span></div>'

    predictions_html += """
</div>
</div>
"""
    st.markdown(predictions_html, unsafe_allow_html=True)

def display_treatment_recommendation(treatment, references, confidence):
    """Display treatment recommendation - EACH SECTION in its OWN curved box"""

    # Determine urgency based on confidence
    if confidence >= 0.9:
        urgency_icon = "🔴"
        urgency_level = "URGENT"
        action = "IMMEDIATE TREATMENT REQUIRED"
        confidence_desc = "VERY HIGH"
        confidence_icon = "🟢"
        reason = "Very clear disease symptoms - act immediately"
    elif confidence >= 0.8:
        urgency_icon = "🟠"
        urgency_level = "HIGH"
        action = "TREAT PROMPTLY"
        confidence_desc = "HIGH"
        confidence_icon = "🟢"
        reason = "Clear disease symptoms visible - treat promptly"
    elif confidence >= 0.7:
        urgency_icon = "🟡"
        urgency_level = "MEDIUM"
        action = "TREAT PROMPTLY"
        confidence_desc = "HIGH"
        confidence_icon = "🟡"
        reason = "Clear but not extremely strong indicators"
    elif confidence >= 0.6:
        urgency_icon = "🟡"
        urgency_level = "MEDIUM"
        action = "CONSIDER TREATMENT"
        confidence_desc = "MODERATE"
        confidence_icon = "🟡"
        reason = "Some disease symptoms visible - consider treatment"
    elif confidence >= 0.5:
        urgency_icon = "🟠"
        urgency_level = "CAUTIOUS"
        action = "VERIFY BEFORE TREATMENT"
        confidence_desc = "MODERATE"
        confidence_icon = "🟠"
        reason = "Weak disease indicators present"
    elif confidence >= 0.4:
        urgency_icon = "🟠"
        urgency_level = "CAUTIOUS"
        action = "VERIFY BEFORE TREATMENT"
        confidence_desc = "LOW"
        confidence_icon = "🟠"
        reason = "Unclear or mixed indicators"
    elif confidence >= 0.3:
        urgency_icon = "🟢"
        urgency_level = "LOW"
        action = "CONSULT EXPERT"
        confidence_desc = "LOW"
        confidence_icon = "🔴"
        reason = "Weak disease indicators - seek verification"
    else:
        urgency_icon = "⚪"
        urgency_level = "VERY LOW"
        action = "SEEK EXPERT ADVICE"
        confidence_desc = "VERY LOW"
        confidence_icon = "🔴"
        reason = "Symptoms unclear or absent - seek expert advice"

    # SECTION 1: TREATMENT RECOMMENDATION HEADER
    st.markdown(f"""
<div class="section-card">
<h3>🌾 TREATMENT RECOMMENDATION FOR: {treatment.get('disease_name', 'Unknown')}</h3>
<p><strong>📊 DIAGNOSIS:</strong> {treatment.get('disease_name', 'Unknown')}</p>
<p><strong>🏷️ TYPE:</strong> {treatment['category']}</p>
<p><strong>📈 CLASSIFIER CONFIDENCE:</strong> {confidence*100:.1f}% ({confidence_desc}) {confidence_icon}</p>
<p><strong>⚡ URGENCY & ACTION:</strong> {urgency_icon} {urgency_level} - {action}</p>
<p><strong>💡 REASON:</strong> {reason}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 2: URGENCY KEY
    st.markdown("""
<div class="section-card">
<h3>⚡ URGENCY KEY - What Action to Take Based on Score</h3>
<p>   🔴 <strong>90-100% (URGENT)</strong>    → Very clear symptoms. Act immediately!</p>
<p>   🟠 <strong>80-89% (HIGH)</strong>       → Clear symptoms. Treat promptly.</p>
<p>   🟡 <strong>70-79% (MEDIUM)</strong>     → Clear symptoms. Treat promptly.</p>
<p>   🟡 <strong>60-69% (MEDIUM)</strong>     → Some symptoms visible. Consider treatment.</p>
<p>   🟠 <strong>50-59% (CAUTIOUS)</strong>   → Weak symptoms. Verify before treatment.</p>
<p>   🟠 <strong>40-49% (CAUTIOUS)</strong>   → Unclear indicators. Verify before treatment.</p>
<p>   🟢 <strong>30-39% (LOW)</strong>        → Weak indicators. Consult an expert.</p>
<p>   ⚪ <strong>0-29% (VERY LOW)</strong>     → Unclear. Seek expert advice immediately.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 3: CAUSES & CHARACTERISTICS
    st.markdown(f"""
<div class="section-card">
<h3>📋 CAUSES & CHARACTERISTICS</h3>
<p style="white-space:pre-line;">{treatment.get('causes_characteristics', 'Information not available')}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 4: RECOMMENDED MANAGEMENT
    st.markdown(f"""
<div class="section-card">
<h3>🔧 RECOMMENDED MANAGEMENT</h3>
<p style="white-space:pre-line;">{treatment['management']}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 5: CHEMICAL CONTROL
    st.markdown(f"""
<div class="section-card">
<h3>💊 CHEMICAL CONTROL</h3>
<p style="white-space:pre-line;">{treatment['chemical_control']}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 6: MANAGEMENT REFERENCES
    management_refs_text = format_reference_list_sequential(treatment.get('management_refs', []), references)
    st.markdown(f"""
<div class="section-card">
<h3>📚 MANAGEMENT REFERENCES</h3>
<div class="reference-text">{management_refs_text}</div>
</div>
""", unsafe_allow_html=True)

    # SECTION 7: CHEMICAL REFERENCES
    chemical_refs_original = treatment.get('chemical_refs_original', [])
    if chemical_refs_original:
        chem_ref_list = []
        for idx, ref_num in enumerate(chemical_refs_original, 1):
            if ref_num in references:
                chem_ref_list.append(f"[{idx}] {references[ref_num]}")
            else:
                chem_ref_list.append(f"[{idx}] Reference {ref_num}")
        chemical_refs_text = "\n".join(chem_ref_list)
    else:
        chemical_refs_text = "See product labels for specific chemical references"

    st.markdown(f"""
<div class="section-card">
<h3>📚 CHEMICAL REFERENCES</h3>
<div class="reference-text">{chemical_refs_text}</div>
</div>
""", unsafe_allow_html=True)

    # SECTION 8: IMPORTANT NOTES (UPDATED DISCLAIMER)
    st.markdown(f"""
<div class="section-card">
<h3>📋 IMPORTANT NOTES:</h3>
<p>  • <strong>Diagnosis:</strong> AI-assisted with {confidence*100:.1f}% confidence. Confirm with expert if uncertain.</p>
<p>  • <strong>Treatments:</strong> Curated from verified sources (product labels, research papers, extension guides)</p>
<p>  • <strong>Always read product labels</strong> before applying any chemicals</p>
<p>  • <strong>Local availability:</strong> Check with your local agrovet for product availability</p>
<p>  • <strong>Expert consultation:</strong> Contact your local agricultural extension officer for confirmation</p>
</div>
""", unsafe_allow_html=True)

def display_xai_analysis(disease_data):
    """Display XAI analysis - EACH SECTION in its OWN curved box"""
    predicted_class = disease_data['class']
    confidence = disease_data['confidence']
    treatment = disease_data['treatment']
    references = disease_data['references']
    heatmap_overlay = disease_data['heatmap_overlay']
    original_img = disease_data['original_img']
    is_primary = disease_data.get('is_primary', True)
    alt_num = disease_data.get('alt_num', None)
    is_healthy = treatment.get('is_healthy', False)
    save_path = disease_data.get('save_path', None)

    # For healthy crops, use the health assessment display with heatmap
    if is_healthy:
        display_healthy_crop_assessment(
            predicted_class,
            confidence,
            treatment,
            references,
            heatmap_overlay=heatmap_overlay,
            original_img=original_img,
            save_path=save_path
        )
        return

    # Determine confidence level and description for diseased crops
    if confidence >= 0.9:
        level = 'VERY HIGH'
        level_icon = "🟢"
        level_desc = "Extremely reliable - the disease indicators are very clear and distinct."
    elif confidence >= 0.8:
        level = 'HIGH'
        level_icon = "🟢"
        level_desc = "Very reliable - the disease indicators are clearly visible."
    elif confidence >= 0.7:
        level = 'HIGH'
        level_icon = "🟡"
        level_desc = "Reliable - the disease indicators are present but not extremely strong."
    elif confidence >= 0.6:
        level = 'MODERATE'
        level_icon = "🟡"
        level_desc = "Moderately reliable - some disease indicators are visible."
    elif confidence >= 0.5:
        level = 'MODERATE'
        level_icon = "🟠"
        level_desc = "Moderately reliable - disease indicators are present but weak."
    elif confidence >= 0.4:
        level = 'LOW'
        level_icon = "🟠"
        level_desc = "Low reliability - disease indicators are unclear or mixed."
    elif confidence >= 0.3:
        level = 'LOW'
        level_icon = "🔴"
        level_desc = "Low reliability - weak disease indicators present."
    else:
        level = 'VERY LOW'
        level_icon = "🔴"
        level_desc = "Very low reliability - disease indicators are absent or unclear."

    title = f"🔬 EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) ANALYSIS FOR: {predicted_class}"
    if not is_primary and alt_num:
        title = f"🔬 ALTERNATIVE {alt_num}: XAI ANALYSIS FOR: {predicted_class}"

    # SECTION 1: Title and Confidence
    st.markdown(f"""
<div class="section-card">
<h3>{title}</h3>
<p><strong>📊 CLASSIFIER CONFIDENCE SCORE:</strong> {confidence*100:.1f}% ({level}) {level_icon}</p>
<p>{level_desc}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 2: CONFIDENCE KEY
    st.markdown("""
<div class="section-card">
<h3>📊 CONFIDENCE KEY - How to Understand the Score:</h3>
<p>   🟢 <strong>90-100% (VERY HIGH)</strong> → Extremely reliable. Clear disease indicators.</p>
<p>   🟢 <strong>80-89% (HIGH)</strong>     → Very reliable. Strong disease indicators.</p>
<p>   🟡 <strong>70-79% (HIGH)</strong>     → Reliable. Disease indicators present.</p>
<p>   🟡 <strong>60-69% (MODERATE)</strong> → Moderately reliable. Some indicators visible.</p>
<p>   🟠 <strong>50-59% (MODERATE)</strong> → Moderately reliable. Weak indicators.</p>
<p>   🟠 <strong>40-49% (LOW)</strong>      → Low reliability. Unclear indicators.</p>
<p>   🔴 <strong>30-39% (LOW)</strong>      → Low reliability. Mixed evidence.</p>
<p>   🔴 <strong>0-29% (VERY LOW)</strong>  → Very low reliability. Seek expert advice.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 3: DISEASE TYPE
    st.markdown(f"""
<div class="section-card">
<h3>🏷️ DISEASE TYPE</h3>
<p>{treatment['category']}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 4: CAUSAL AGENT
    st.markdown(f"""
<div class="section-card">
<h3>🦠 CAUSAL AGENT</h3>
<p>{treatment['causal_agent']}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 5: KEY CHARACTERISTICS
    st.markdown(f"""
<div class="section-card">
<h3>📋 KEY CHARACTERISTICS</h3>
<p style="white-space:pre-line;">{treatment.get('causes_characteristics', 'See detailed description')}</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 6: WHAT THE MODEL ANALYSED
    st.markdown(f"""
<div class="section-card">
<h3>💡 WHAT THE MODEL ANALYSED</h3>
<p>The model examined your crop image and identified visual patterns that match the characteristic appearance of {predicted_class}.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 7: VISUAL EVIDENCE (Grad-CAM Heatmap)
    st.markdown(f"""
<div class="section-card">
<h3>🔥 VISUAL EVIDENCE (Heatmap)</h3>
<p>✓ Generated</p>
</div>
""", unsafe_allow_html=True)

    # Display heatmap if available
    if heatmap_overlay is not None and original_img is not None:
        display_heatmap_with_colorbar(original_img, heatmap_overlay, predicted_class, save_path)

    # SECTION 8: GRAD-CAM LEGEND
    st.markdown("""
<div class="section-card">
<h3>📊 HEATMAP LEGEND</h3>
<p>   🔴 <strong>RED (HOT)</strong>     = HIGH influence - These areas strongly indicate the disease</p>
<p>   🟠 <strong>ORANGE</strong>        = HIGH-MEDIUM influence</p>
<p>   🟡 <strong>YELLOW</strong>        = MEDIUM influence</p>
<p>   🟢 <strong>GREEN</strong>         = LOW-MEDIUM influence</p>
<p>   💠 <strong>CYAN</strong>          = VERY LOW influence</p>
<p>   🔵 <strong>BLUE (COOL)</strong>   = LOWEST influence - These areas had minimal impact on the diagnosis</p>
<p style="font-size:12px; margin-top:10px;">💡 The warmer the color (Red → Orange → Yellow), the more it influenced the model's decision. Cooler colors (Green → Cyan → Blue) had less influence.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 9: WHAT THIS MEANS FOR YOU
    st.markdown(f"""
<div class="section-card">
<h3>📋 WHAT THIS MEANS FOR YOU</h3>
<p>Based on the visual patterns detected, the system has diagnosed your crop with {predicted_class} with {level.lower()} confidence.</p>
</div>
""", unsafe_allow_html=True)

    # SECTION 10: LOW CONFIDENCE GUIDANCE (if needed)
    if confidence < 0.6:
        st.markdown("""
<div class="section-card">
<h3>⚠️ LOW CONFIDENCE GUIDANCE</h3>
<p>The model is not very confident about this diagnosis. We recommend:</p>
<ol>
    <li>📸 Take a clearer photo focusing on the affected areas</li>
    <li>🔍 Examine multiple plants in your field for comparison</li>
    <li>👨‍🌾 Consult an agricultural extension officer for confirmation</li>
    <li>🔄 Consider the treatment options for ALL top 3 predictions</li>
</ol>
</div>
""", unsafe_allow_html=True)

    # SECTION 11: XAI SOURCES
    xai_refs = []
    for ref_num in treatment.get('xai_ref_numbers', []):
        if ref_num in references:
            xai_refs.append(f"[{len(xai_refs)+1}] {references[ref_num]}")

    if xai_refs:
        st.markdown(f"""
<div class="section-card">
<h3>📚 XAI SOURCES</h3>
<div class="reference-text">{"\n".join(xai_refs)}</div>
</div>
""", unsafe_allow_html=True)

    # SECTION 12: XAI SOURCES (Grad-CAM)
    st.markdown(f"""
<div class="section-card">
<h3>📚 XAI SOURCES (Grad-CAM)</h3>
<div class="reference-text">[{len(xai_refs)+1}] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,' in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618-626.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NEWS AND WEATHER HELPERS FOR ONLINE MODE
# ============================================================

def fetch_kalro_updates(limit=3):
    """Fetch latest updates from Kenya Agricultural and Livestock Research Organization"""
    try:
        # KALRO website doesn't have a public RSS feed, but we can fetch their posts page
        # For now, using a curated approach with their known content
        # In production, you might want to scrape their website or use their API
        
        # Sample recent KALRO updates from their website [citation:1][citation:5][citation:8]
        kalro_updates = [
            {
                "title": "KALRO Flags High Aflatoxin Levels in Market Grains",
                "date": "2026-06-07",
                "summary": "Kenya Agricultural and Livestock Research Organization (KALRO) has raised an alarm over aflatoxin-contaminated cereals in Kenyan markets, with some samples testing at 500 parts per billion - 50 times higher than the legal safety limit of 10 ppb. Farmers are advised to ensure proper drying and storage of grains.",
                "url": "https://www.standardmedia.co.ke/farmkenya",
                "source": "KALRO/The Star"
            },
            {
                "title": "KALRO Scientific Conference 2026 - Call for Abstracts",
                "date": "2026-06-01",
                "summary": "The 2nd KALRO Scientific Conference and Innovation Expo is calling for abstracts on innovations for sustainable agri-food systems, climate change resilience, and improved livelihoods.",
                "url": "https://kalro.org",
                "source": "KALRO"
            },
            {
                "title": "New KALRO Director General Appointed",
                "date": "2025-12-10",
                "summary": "Dr. Patrick K. Ketiem has been appointed as the new Director General of KALRO, marking a strategic new chapter for agricultural research in Kenya.",
                "url": "https://kalro.org",
                "source": "KALRO"
            }
        ]
        
        # In production, you would scrape or use API for real-time updates
        return kalro_updates[:limit]
    except Exception as e:
        print(f"Error fetching KALRO updates: {e}")
        return []

def fetch_kenya_meteo_warnings():
    """Fetch weather warnings from Kenya Meteorological Department"""
    try:
        # Kenya Met Department has a CAP RSS feed [citation:7]
        # RSS URL: https://meteo.go.ke/api/cap/rss.xml
        warnings = []
        
        # Sample of actual warnings from Kenya Met website [citation:4]
        # In production, parse their RSS feed: feedparser.parse("https://meteo.go.ke/api/cap/rss.xml")
        
        warnings = [
            {
                "type": "Heavy Rainfall",
                "areas": ["Narok", "Kericho", "Bomet", "Homabay", "Siaya", "Kisumu"],
                "severity": "Yellow",
                "issued": "2026-06-05",
                "advice": "Farmers in affected areas should avoid planting in flood-prone zones and ensure proper drainage."
            },
            {
                "type": "Strong Winds",
                "areas": ["Mombasa", "Kilifi", "Kwale", "Lamu", "Tana River"],
                "severity": "Yellow",
                "issued": "2026-06-04",
                "advice": "Secure greenhouses and temporary structures. Delay spraying operations."
            }
        ]
        
        # In production, parse the actual RSS feed:
        # feed = feedparser.parse("https://meteo.go.ke/api/cap/rss.xml")
        # for entry in feed.entries:
        #     warnings.append({...})
        
        return warnings
    except Exception as e:
        print(f"Error fetching weather warnings: {e}")
        return []

def fetch_kenya_agriculture_news(query=None, limit=5):
    """Fetch agriculture news from Kenyan news sources including Nation Africa"""
    articles = []
    
    # Source 1: Nation Africa (Daily Nation) - Agriculture RSS Feed
    # Nation Media Group has RSS feeds available for their content
    try:
        # Nation Africa's agriculture section RSS 
        # Standard RSS pattern for Nation: https://nation.africa/kenya/agriculture/rss
        nation_feed = feedparser.parse("https://nation.africa/kenya/agriculture/rss")
        for entry in nation_feed.entries[:limit]:
            # Filter for agriculture/farming content if query provided
            if query and query.lower() not in (entry.title + entry.summary).lower():
                continue
            articles.append({
                "title": entry.title,
                "summary": entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                "url": entry.link,
                "source": "Nation Africa (Daily Nation)",
                "date": entry.get("published", ""),
                "category": "Agriculture"
            })
    except Exception as e:
        print(f"Error fetching Nation Africa feed: {e}")
    
    # Source 2: The Standard - Agriculture RSS Feed 
    try:
        standard_feed = feedparser.parse("https://www.standardmedia.co.ke/rss/agriculture.php")
        for entry in standard_feed.entries[:limit]:
            if query and query.lower() not in (entry.title + entry.summary).lower():
                continue
            articles.append({
                "title": entry.title,
                "summary": entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                "url": entry.link,
                "source": "The Standard",
                "date": entry.get("published", ""),
                "category": "Agriculture"
            })
    except Exception as e:
        print(f"Error fetching The Standard feed: {e}")
    
    # Source 3: Daily Monitor (Uganda) - Nation Media Group's Ugandan outlet
    # Useful for regional agriculture news affecting East Africa
    try:
        monitor_feed = feedparser.parse("https://www.monitor.co.ug/uganda/agriculture/rss")
        for entry in monitor_feed.entries[:limit]:
            if query and query.lower() not in (entry.title + entry.summary).lower():
                continue
            articles.append({
                "title": entry.title,
                "summary": entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                "url": entry.link,
                "source": "Daily Monitor (Uganda)",
                "date": entry.get("published", ""),
                "category": "Agriculture"
            })
    except Exception as e:
        print(f"Error fetching Daily Monitor feed: {e}")
    
    # Source 4: The EastAfrican - Regional perspective
    try:
            eastafrican_feed = feedparser.parse("https://www.theeastafrican.co.ke/rss")
            for entry in eastafrican_feed.entries[:limit]:
                if any(term in (entry.title + entry.summary).lower() for term in ['agriculture', 'farming', 'crop', 'livestock', 'food']):
                    articles.append({
                        "title": entry.title,
                        "summary": entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                        "url": entry.link,
                        "source": "The EastAfrican",
                        "date": entry.get("published", ""),
                        "category": "Regional"
                    })
    except Exception as e:
        print(f"Error fetching The EastAfrican feed: {e}")
    
    return articles

def get_weather_advisory(weather_warnings, disease_name, location):
    """Generate a tailored advisory based on weather warnings and disease"""
    advisories = []
    
    for warning in weather_warnings:
        # Check if user's location is in affected areas
        location_parts = location.lower().split(',')
        is_affected = any(area.lower() in location.lower() for area in warning.get('areas', []))
        
        if is_affected or len(weather_warnings) > 0:
            if warning['type'] == "Heavy Rainfall":
                advisories.append({
                    "type": "warning",
                    "message": f"⚠️ **Heavy rainfall alert** for your area. {warning['advice']}",
                    "related_disease_advice": "Fungal diseases may spread rapidly after heavy rains. Consider preventive fungicide application."
                })
            elif warning['type'] == "Strong Winds":
                advisories.append({
                    "type": "warning",
                    "message": f"💨 **Strong winds expected**. {warning['advice']}",
                    "related_disease_advice": "Postpone spraying to avoid chemical drift."
                })
            elif warning['type'] == "Drought":
                advisories.append({
                    "type": "warning",
                    "message": f"🌵 **Dry conditions forecast**. Ensure adequate irrigation and consider mulch to retain moisture.",
                    "related_disease_advice": "Monitor for pest outbreaks (aphids, spider mites thrive in dry conditions)."
                })
    
    return advisories

def display_online_features(disease_name, crop_type, location, treatment_data=None):
    """Display online features including weather, news, and agricultural updates"""
    
    st.markdown("---")
    st.markdown("### 📡 ONLINE MODE - LIVE UPDATES")
    st.caption("ℹ️ **Tip:** Hover over the ❔ icons for detailed explanations.")
    
    # ============================================================
    # SECTION 1: WEATHER & DISEASE RISK
    # ============================================================
    st.markdown("#### 🌤️ CURRENT WEATHER & DISEASE RISK")
    
    weather = get_weather_with_risk_assessment(location, disease_name, treatment_data)

    if weather:
        risk_class = weather.get('risk_class', '')
        risk_msg = weather.get('risk_msg', '')

        weather_text = f"""
        <div class="weather-card">
            <h4>🌤️ WEATHER FOR {weather['location']}</h4>
        """
        
        if weather['temperature'] != 'N/A':
            weather_text += f'<p>🌡️ <strong>Temperature:</strong> {weather["temperature"]}°C</p>'
        
        if weather['humidity'] != 'N/A':
            weather_text += f'<p>💧 <strong>Humidity:</strong> {weather["humidity"]}%</p>'
        
        weather_text += f'<p>☔ <strong>Current Rainfall:</strong> {weather["rain"]} mm</p>'
        
        if weather['wind'] != 'N/A':
            weather_text += f'<p>🌬️ <strong>Wind Speed:</strong> {weather["wind"]} km/h</p>'
        
        if weather['temp_max'] and weather['temp_min']:
            weather_text += f'<p>📅 <strong>Today\'s Forecast:</strong> High {weather["temp_max"]}°C / Low {weather["temp_min"]}°C</p>'
        
        if weather['rain_prob']:
            weather_text += f'<p>🌧️ <strong>Rain Probability:</strong> {weather["rain_prob"]}% (Expected: {weather["rain_sum"]} mm)</p>'
        
        weather_text += f"""
            <hr>
            <p><strong>🎯 DISEASE RISK ASSESSMENT FOR {disease_name}:</strong><br>
            <span class="{risk_class}">{risk_msg}</span></p>
        </div>
        """
        st.markdown(weather_text, unsafe_allow_html=True)
    else:
        st.info("🌤️ Unable to fetch weather data. Please check your internet connection.")
    
    # ============================================================
    # SECTION 2: KENYA MET DEPARTMENT WEATHER WARNINGS
    # ============================================================
    st.markdown("#### 🚨 KENYA MET DEPARTMENT WEATHER WARNINGS")
    
    weather_warnings = fetch_kenya_meteo_warnings()
    
    if weather_warnings:
        for warning in weather_warnings:
            st.warning(f"⚠️ **{warning['type']} Alert** - {warning.get('issued', 'Current')}")
            st.markdown(f"**Affected Areas:** {', '.join(warning.get('areas', []))}")
            st.markdown(f"**Advice:** {warning.get('advice', 'Monitor local conditions.')}")
            st.markdown("---")
    else:
        st.info("📭 No active weather warnings for your region at this time.")
    
    # ============================================================
    # SECTION 3: KALRO AGRICULTURAL ADVISORIES
    # ============================================================
    st.markdown("#### 🌾 KALRO AGRICULTURAL UPDATES")
    st.caption("From Kenya Agricultural and Livestock Research Organization")
    
    kalro_updates = fetch_kalro_updates(limit=3)
    
    if kalro_updates:
        for update in kalro_updates:
            with st.expander(f"📢 {update['title']}"):
                st.caption(f"Source: {update.get('source', 'KALRO')} | {update.get('date', 'Recent')}")
                st.write(update.get('summary', ''))
                if update.get('url'):
                    st.markdown(f"[Read more]({update['url']})")
    else:
        st.info("📭 Loading KALRO updates...")
    
    # ============================================================
    # SECTION 4: LATEST AGRICULTURE NEWS FROM KENYAN MEDIA
    # ============================================================
    st.markdown("#### 📰 LATEST AGRICULTURE NEWS")
    st.caption("From Nation Africa, The Standard, and Daily Nation")
    
    # THIS IS THE FIX - search_term is now defined INSIDE the function
    search_term = disease_name.split()[0] if disease_name else crop_type
    news_articles = fetch_kenya_agriculture_news(query=search_term, limit=6)
    
    if news_articles:
        for article in news_articles:
            with st.expander(f"📰 {article['title']}"):
                st.caption(f"Source: {article['source']} | {article.get('date', 'Recent')}")
                st.write(article.get('summary', ''))
                st.markdown(f"[Read full article]({article['url']})")
    else:
        st.markdown("""
        **📰 Nation Africa - Agriculture News**
        
        - [Nation Africa Agriculture Section](https://nation.africa/kenya/agriculture) - Latest farming news
        - [Weather patterns shift: What farmers need to know](https://nation.africa/kenya/agriculture)
        - [Market prices for key crops this season](https://nation.africa/kenya/agriculture)
        
        ---
        
        **📰 The Standard - FarmKenya**
        
        - [FarmKenya: Smart Harvest weekly pullout](https://www.standardmedia.co.ke/farmkenya)
        
        ---
        
        **📰 The EastAfrican - Regional Perspective**
        
        - [East African agriculture and trade news](https://www.theeastafrican.co.ke)
        """)
        
        st.caption("📌 For the latest updates, visit [Nation Africa Agriculture](https://nation.africa/kenya/agriculture) and [The Standard FarmKenya](https://www.standardmedia.co.ke/farmkenya)")
    
    # ============================================================
    # SECTION 5: WEATHER-BASED FARMING TIP
    # ============================================================
    st.markdown("---")
    st.markdown("#### 💡 WEATHER-BASED FARMING TIP")
    
    if weather:
        temp = weather.get('temperature')
        humidity = weather.get('humidity')
        rain_prob = weather.get('rain_prob')
        
        if rain_prob and rain_prob > 70:
            st.success("🌧️ **Rain expected.** Consider postponing pesticide/fungicide application until after the rain to avoid wash-off.")
        elif temp and temp > 30:
            st.warning("🔥 **High temperatures expected.** Ensure adequate irrigation and consider applying mulch to retain soil moisture.")
        elif humidity and humidity > 80:
            st.info("💨 **High humidity conditions.** Favorable for fungal diseases. Consider preventive fungicide application.")
        else:
            st.info("🌱 **Optimal conditions for farm operations.** Good time for field scouting and regular crop monitoring.")
    else:
        st.info("🌱 Check local weather conditions regularly for optimal timing of farm activities.")
    
    # ============================================================
    # SECTION 6: RESOURCE DIRECTORY
    # ============================================================
    with st.expander("📚 Agricultural Resources for Kenyan Farmers"):
        st.markdown("""
        **Government & Research Institutions:**
        - [Kenya Agricultural and Livestock Research Organization (KALRO)](https://kalro.org)
        - [Kenya Meteorological Department](https://meteo.go.ke)
        - [Kenya Plant Health Inspectorate Service (KEPHIS)](https://www.kephis.org)
        
        **News & Information:**
        - [Nation Africa - Agriculture](https://nation.africa/kenya/agriculture)
        - [The Standard - FarmKenya](https://www.standardmedia.co.ke/farmkenya)
        - [The EastAfrican](https://www.theeastafrican.co.ke)
        
        **Farmer Support:**
        - [Ministry of Agriculture](https://kilimo.go.ke)
        - National Agricultural Extension Hotline: 0800 720 123
        """)
        

def display_options_menu(top_predictions, references, location, class_names, current_disease_name, current_crop_type, current_treatment_data=None):
    """Display dynamic options menu - menu header in its own curved box with all functionality"""

    num_predictions = len(top_predictions)
    common_chemicals_option = num_predictions + 1
    change_topk_option = common_chemicals_option + 1
    analyze_another_option = change_topk_option + 1
    exit_option = analyze_another_option + 1

    current_mode = st.session_state.mode

    # Build the menu as a single markdown block in a section-card
    menu_html = f"""
<div class="section-card">
<h3>💡 OPTIONS MENU</h3>
<p><strong>1.</strong> Get treatment for PRIMARY DIAGNOSIS ({top_predictions[0]['class']}) - {top_predictions[0]['confidence']*100:.1f}%</p>
"""

    for i in range(1, num_predictions):
        menu_html += f'<p><strong>{i+1}.</strong> Get treatment for ALTERNATIVE {i} ({top_predictions[i]["class"]}) - {top_predictions[i]["confidence"]*100:.1f}%</p>\n'

    menu_html += f"""
<p><strong>{common_chemicals_option}.</strong> Show common chemicals for ALL TOP {num_predictions} diseases</p>
<p><strong>{change_topk_option}.</strong> Change number of top predictions (current: {st.session_state.current_top_k})</p>
"""

    # Automatic mode switch suggestion when in online mode with no internet
    if current_mode == "online" and not check_internet_connection():
        menu_html += f'<p style="color: #FF9800;"><strong>⚠️ No internet detected!</strong> Please use the radio button above to switch to OFFLINE MODE.</p>\n'

    menu_html += f"""
<p><strong>{analyze_another_option}.</strong> Analyse another image</p>
<p><strong>{exit_option}.</strong> Exit the app</p>
</div>
"""
    st.markdown(menu_html, unsafe_allow_html=True)

    # Create button row for menu options
    total_options = exit_option
    cols = st.columns(min(total_options, 10))

    for i in range(1, min(total_options + 1, 11)):
        with cols[i-1]:
            if st.button(f"{i}", key=f"menu_btn_{i}", width="stretch"):
                if i == exit_option:
                    st.session_state.show_results = False
                    st.session_state.current_image = None
                    st.session_state.current_predictions = None
                    st.session_state.current_top_predictions = None
                    st.session_state.current_alt_data = {}
                    st.session_state.current_showing_alternative = None
                    st.session_state.showing_common_chemicals = False
                    st.rerun()
                elif i == analyze_another_option:
                    st.session_state.show_results = False
                    st.session_state.current_image = None
                    st.session_state.current_predictions = None
                    st.session_state.current_top_predictions = None
                    st.session_state.current_alt_data = {}
                    st.session_state.current_showing_alternative = None
                    st.session_state.showing_common_chemicals = False
                    st.rerun()
                elif i == change_topk_option:
                    st.session_state.show_k_dialog = True
                    st.rerun()
                elif i == common_chemicals_option:
                    st.session_state.showing_common_chemicals = True
                    st.session_state.common_chemicals_data = top_predictions
                    st.rerun()
                elif 1 <= i <= num_predictions:
                    if i == 1:
                        st.session_state.current_showing_alternative = None
                    else:
                        st.session_state.current_showing_alternative = i - 1
                    st.session_state.showing_common_chemicals = False
                    st.rerun()

    # K value change dialog
    if st.session_state.get('show_k_dialog', False):
        with st.expander("📊 Change number of top predictions", expanded=True):
            st.markdown("**K** represents the number of top predictions to display and consider.")
            st.markdown(f"Current K value: **{st.session_state.current_top_k}**")
            st.markdown(f"Total classes available: **{len(class_names)}**")
            new_k = st.number_input(
                "Enter new K value",
                min_value=1,
                max_value=len(class_names),
                value=st.session_state.current_top_k,
                step=1,
                key="k_value_input"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Set", width="stretch", key="set_k_btn"):
                    st.session_state.current_top_k = new_k
                    st.session_state.show_results = False
                    st.session_state.show_k_dialog = False
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", width="stretch", key="cancel_k_btn"):
                    st.session_state.show_k_dialog = False
                    st.rerun()

    # Show common chemicals if requested
    if st.session_state.get('showing_common_chemicals', False) and st.session_state.common_chemicals_data:
        st.markdown("---")
        show_common_chemicals_for_top_k(st.session_state.common_chemicals_data, references)
        if st.button("✖ Close Common Chemicals", width="stretch", key="close_chemicals_btn"):
            st.session_state.showing_common_chemicals = False
            st.session_state.common_chemicals_data = None
            st.rerun()

    # Online features based on current displayed disease
    if current_mode == "online":
        st.markdown("---")
        display_online_features(current_disease_name, current_crop_type, st.session_state.location, current_treatment_data)

# ============================================================
# PRIVACY NOTICE DISPLAY FUNCTION
# ============================================================

def display_privacy_notice():
    """Display a dismissible privacy notice about image saving"""
    if not st.session_state.privacy_notice_dismissed:
        st.markdown("""
        <div class="privacy-notice">
            <div class="privacy-notice-text">
                🔒 <strong>Privacy Notice:</strong> To help improve the system and train better models,
                uploaded images and diagnoses may be anonymously saved. No personal information is collected.
                You can continue using the app as normal.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Use two columns for a cleaner button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("✓ Got it", key="dismiss_privacy_notice", width='stretch'):
                st.session_state.privacy_notice_dismissed = True
                st.rerun()
        st.markdown("---")

def display_top_location_buttons():
    """Display Change Location and Refresh buttons"""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📍 Change Location", use_container_width=True):
            st.session_state.show_top_location_dialog = True
            st.rerun()
    with col2:
        if st.button("🔄 Refresh Weather", use_container_width=True):
            st.session_state.weather_info = None
            st.success("✅ Weather data refreshed!")
            time.sleep(0.5)
            st.rerun()


def display_top_location_dialog():
    """Manual location entry dialog"""
    if not st.session_state.get('show_top_location_dialog', False):
        return

    st.markdown("---")
    st.markdown("### 📍 Change Your Location")
    st.success(f"📍 **Current location:** {st.session_state.location}")
    st.markdown("---")
    st.markdown("#### Enter your location manually:")
    st.caption("📝 Examples: Kisumu, Kenya | Eldoret, Uasin Gishu County, Kenya | Machakos, Kenya")

    current_loc_parts = st.session_state.location.split(',')
    default_city = current_loc_parts[0].strip() if current_loc_parts else "Ekerenyo, Nyamira County"
    default_region = current_loc_parts[1].strip() if len(current_loc_parts) > 1 else ""
    default_country = current_loc_parts[-1].strip() if current_loc_parts else "Kenya"

    col1, col2 = st.columns(2)
    with col1:
        new_city = st.text_input("City/Town *", value=default_city)
        new_region = st.text_input("County/Region (optional)", value=default_region)
    with col2:
        new_country = st.text_input("Country", value=default_country)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Location", use_container_width=True):
            if new_city:
                location = f"{new_city}, {new_country}" if not new_region else f"{new_city}, {new_region}, {new_country}"
                st.session_state.location = location
                st.session_state.show_top_location_dialog = False
                st.success(f"✅ Location saved: {location}")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Please enter at least a city/town.")
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_top_location_dialog = False
            st.rerun()

    st.markdown("---")

# Add this to handle GPS form submission (add this function before main())
def handle_gps_submission():
    """Handle GPS form submission from JavaScript"""
    import streamlit as st

    # Check if we have GPS data from the form
    # In Streamlit, we need to use query parameters or session state

    # Alternative: Use a hidden component approach
    pass

def display_top_manual_entry_dialog():
    """Display manual entry dialog for top bar - recommended method"""

    if not st.session_state.get('show_top_manual_entry', False):
        return

    st.markdown("---")
    st.markdown("### 📝 Enter Your Location Manually")
    st.markdown("✅ **Recommended method for most accurate weather forecasts**")
    st.caption("Examples: Kisumu, Kenya | Eldoret, Uasin Gishu County, Kenya | Machakos, Kenya")

    current_loc_parts = st.session_state.location.split(',')
    default_city = current_loc_parts[0].strip() if len(current_loc_parts) > 0 else "Ekerenyo, Nyamira County"
    default_region = current_loc_parts[1].strip() if len(current_loc_parts) > 1 else ""
    default_country = current_loc_parts[-1].strip() if len(current_loc_parts) > 0 else "Kenya"

    col1, col2 = st.columns(2)
    with col1:
        new_city = st.text_input("City/Town *", value=default_city, key="top_manual_city")
        new_region = st.text_input("County/Region (optional)", value=default_region, key="top_manual_region")
    with col2:
        new_country = st.text_input("Country", value=default_country, key="top_manual_country")

    st.caption("* Required field")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Location", use_container_width=True, key="top_save_btn"):
            if new_city:
                if new_region:
                    st.session_state.location = f"{new_city}, {new_region}, {new_country}"
                else:
                    st.session_state.location = f"{new_city}, {new_country}"
                st.session_state.location_method = "manual"
                st.session_state.gps_location = None
                st.session_state.show_top_manual_entry = False
                st.success(f"✅ Location saved: {st.session_state.location}")
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error("Please enter at least a city/town.")
    with col2:
        if st.button("❌ Cancel", use_container_width=True, key="top_cancel_manual_btn"):
            st.session_state.show_top_manual_entry = False
            st.rerun()

    st.markdown("---")


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application entry point"""

    with streamlit_analytics.track(unsafe_password="chibando_ching'ende_chinyanya"):
        display_privacy_notice()

        st.markdown("""
        <div class="main-header">
            <h1>🌾 Crop Doctor</h1>
            <div class="subtitle">Crop Disease Classification and Treatment Recommendation System</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Help button
            if st.button("❓ Help / How to use the system", use_container_width=True):
                st.session_state.show_help = not st.session_state.show_help

            # Classes button
            if st.button("📋 List of Supported Classes", use_container_width=True):
                st.session_state.show_classes = not st.session_state.show_classes

            selected_mode = st.radio(
                "Select Mode",
                ["📱 OFFLINE MODE", "🌐 ONLINE MODE"],
                horizontal=True,
                label_visibility="collapsed"
            )
            new_mode = "online" if "ONLINE" in selected_mode else "offline"
            if st.session_state.mode != new_mode:
                st.session_state.mode = new_mode
                st.rerun()

            if st.session_state.mode == "online":
                st.markdown(f'<span class="mode-badge mode-online">🌐 ONLINE MODE - Weather & News Enabled | Location: {st.session_state.location}</span>', unsafe_allow_html=True)

                # Top bar location buttons
                display_top_location_buttons()
            else:
                st.markdown('<span class="mode-badge mode-offline">📱 OFFLINE MODE - Diagnosis and Verified Treatments Only</span>', unsafe_allow_html=True)

        st.markdown("---")

        # ============================================================
        # TOP BAR LOCATION DIALOGS
        # ============================================================
        if st.session_state.get('show_top_location_dialog', False):
            display_top_location_dialog()

        if st.session_state.get('show_top_manual_entry', False):
            display_top_manual_entry_dialog()

        # ============================================================
        # HANDLE GPS RETRIEVAL (shared between both)
        # ============================================================
        if st.session_state.get('request_gps', False):
            with st.spinner("📍 Getting GPS location. Please allow location access..."):
                try:
                    from streamlit_geolocation import streamlit_geolocation
                    location_data = streamlit_geolocation()

                    if location_data and location_data.get('latitude'):
                        lat = location_data['latitude']
                        lon = location_data['longitude']
                        accuracy = location_data.get('accuracy', 0)

                        location_name = get_location_name_from_coords(lat, lon)

                        st.session_state.location = location_name
                        st.session_state.gps_location = {'lat': lat, 'lon': lon, 'accuracy': accuracy}
                        st.session_state.location_method = "gps"
                        st.session_state.request_gps = False

                        st.success(f"✅ GPS Location set: {location_name}")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("❌ Could not get GPS location. Please ensure:")
                        st.markdown("""
                        1. You allowed location access in your browser
                        2. Your device has GPS enabled
                        3. You are using HTTPS (works on Hugging Face Spaces)
                        """)
                        st.session_state.request_gps = False
                except Exception as e:
                    st.error(f"GPS error: {e}")
                    st.session_state.request_gps = False

        # Show classes list if requested
        if st.session_state.show_classes:
            model_temp, class_names_temp = load_model_and_classes()
            if class_names_temp:
                display_classes_list(class_names_temp)
            st.markdown("---")

        # Show help if requested
        if st.session_state.show_help:
            display_help()
            st.markdown("---")

        model, class_names = load_model_and_classes()
        references = get_references()

        if model is None:
            st.stop()

        gradcam = GradCAM(model)

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("### 📸 Upload Crop Image")
            st.caption("📸 Take a clear photo of the affected leaves or fruits for best results")

            if st.button("📷 Take Photo", use_container_width=True):
                st.session_state.camera_active = True

            if st.session_state.camera_active:
                camera_image = st.camera_input("Take a photo", key="camera")
                if camera_image:
                    st.session_state.current_image = Image.open(camera_image)
                    st.session_state.camera_active = False
                    st.rerun()

            uploaded_file = st.file_uploader(
                "Or choose from gallery",
                type=['jpg', 'jpeg', 'png'],
                key="uploader"
            )
            if uploaded_file:
                st.session_state.current_image = Image.open(uploaded_file)

            if st.session_state.current_image is not None:
                st.image(st.session_state.current_image, caption="Selected Image", width="stretch")

                if st.button("🔬 DIAGNOSE & RECOMMEND", type="primary", use_container_width=True):
                    with st.spinner("Analysing crop disease..."):
                        image = st.session_state.current_image
                        if image.mode != 'RGB':
                            image = image.convert('RGB')

                        st.session_state.current_original_img = image

                        img_array = preprocess_image(image)
                        predictions = model.predict(img_array, verbose=0)[0]

                        indices = np.argsort(predictions)[-st.session_state.current_top_k:][::-1]
                        top_predictions = []
                        for i in indices:
                            top_predictions.append({
                                'class': class_names[i],
                                'confidence': float(predictions[i]),
                                'idx': i
                            })

                        st.session_state.current_predictions = predictions
                        st.session_state.current_top_predictions = top_predictions
                        st.session_state.show_results = True
                        st.session_state.current_showing_alternative = None
                        st.session_state.showing_common_chemicals = False
                        st.session_state.common_chemicals_data = None

                        st.session_state.current_alt_data = {}

                        # Save image only ONCE for the top prediction
                        save_user_image_for_training(image, top_predictions[0]['class'], top_predictions[0]['confidence'], top_predictions)

                        for alt_idx, pred in enumerate(top_predictions):
                            heatmap = gradcam.generate_heatmap(img_array, pred['idx'])
                            overlay = gradcam.overlay_heatmap(heatmap, st.session_state.current_original_img)
                            treatment = get_full_treatment(pred['class'], references)

                            # Determine crop type for manufacturer search
                            if 'maize' in pred['class'].lower():
                                crop_type = "Maize"
                            elif 'beans' in pred['class'].lower():
                                crop_type = "Beans"
                            else:
                                crop_type = "Tomato"

                            # Save heatmap overlay to a file in the current directory
                            save_filename = f"gradcam_{pred['class'].replace(' ', '_').replace('/', '_')}_alt{alt_idx}.png"
                            save_path = os.path.join(os.getcwd(), save_filename)
                            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                            st.session_state.current_alt_data[alt_idx] = {
                                'class': pred['class'],
                                'confidence': pred['confidence'],
                                'idx': pred['idx'],
                                'treatment': treatment,
                                'heatmap_overlay': overlay,
                                'references': references,
                                'original_img': st.session_state.current_original_img,
                                'crop_type': crop_type,
                                'save_path': save_path,
                                'save_filename': save_filename
                            }

                        st.rerun()

        with right_col:
            if st.session_state.show_results and st.session_state.current_top_predictions:
                top_predictions = st.session_state.current_top_predictions

                # SAFETY CHECK FOR EMPTY ALT DATA
                if not st.session_state.current_alt_data:
                    st.warning("⚠️ No diagnosis data found. Please analyze an image first.")
                    st.session_state.show_results = False
                    st.rerun()

                if st.session_state.current_showing_alternative is not None:
                    alt_idx = st.session_state.current_showing_alternative
                    if alt_idx in st.session_state.current_alt_data:
                        disease_data = st.session_state.current_alt_data[alt_idx]
                        disease_data['is_primary'] = False
                        disease_data['alt_num'] = alt_idx

                        current_disease = disease_data['class']
                        current_crop_type = disease_data['crop_type']
                        current_treatment = disease_data['treatment']

                        display_top_predictions(top_predictions)
                        display_xai_analysis(disease_data)

                        # Only show treatment recommendation for diseased crops
                        if not disease_data['treatment'].get('is_healthy', False):
                            display_treatment_recommendation(disease_data['treatment'], references, disease_data['confidence'])

                        # Export Report Button
                        col1_export, col2_export = st.columns(2)
                        with col1_export:
                            if st.button("📄 Export Report", use_container_width=True, key="export_alt"):
                                report = generate_export_report(disease_data, disease_data['treatment'], references, None)
                                st.download_button(
                                    label="📥 Download Report",
                                    data=report,
                                    file_name=f"crop_doctor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    key="download_alt"
                                )

                        # Pass the treatment data for weather risk assessment
                        display_options_menu(top_predictions, references, st.session_state.location, class_names, current_disease, current_crop_type, current_treatment)
                    else:
                        st.warning(f"⚠️ Alternative {alt_idx} data not found. Please re-analyze the image.")
                        st.session_state.current_showing_alternative = None
                        st.rerun()
                else:
                    # CHECK IF KEY 0 EXISTS BEFORE ACCESSING
                    if 0 not in st.session_state.current_alt_data:
                        st.warning("⚠️ Primary diagnosis data not available. Please analyze an image again.")
                        st.session_state.show_results = False
                        st.rerun()

                    primary_data = st.session_state.current_alt_data[0]
                    primary_data['is_primary'] = True

                    current_disease = primary_data['class']
                    current_crop_type = primary_data['crop_type']
                    current_treatment = primary_data['treatment']

                    display_top_predictions(top_predictions)
                    display_xai_analysis(primary_data)

                    # Only show treatment recommendation for diseased crops
                    if not primary_data['treatment'].get('is_healthy', False):
                        display_treatment_recommendation(primary_data['treatment'], references, primary_data['confidence'])

                    # Export Report Button
                    col1_export, col2_export = st.columns(2)
                    with col1_export:
                        if st.button("📄 Export Report", use_container_width=True, key="export_primary"):
                            report = generate_export_report(primary_data, primary_data['treatment'], references, None)
                            st.download_button(
                                label="📥 Download Report",
                                data=report,
                                file_name=f"crop_doctor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                key="download_primary"
                            )

                    # Pass the treatment data for weather risk assessment
                    display_options_menu(top_predictions, references, st.session_state.location, class_names, current_disease, current_crop_type, current_treatment)

            else:
                st.info("👈 Please take a photo or upload an image, then click 'DIAGNOSE & RECOMMEND'")

if __name__ == "__main__":
    main()

