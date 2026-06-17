"""
Crop Doctor - Crop Disease Classification and Treatment Recommendation System
Run: streamlit run streamlit_app.py

Version: 1.3 - Added user upload saving for model improvement with privacy notice
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
import pytz
import hashlib
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone as dt_timezone
from PIL import Image
from matplotlib.patches import Rectangle


def get_local_timestamp():
    """Get current timestamp in Kenya/East Africa time for filenames"""
    eat_timezone = dt_timezone(timedelta(hours=3))
    local_now = datetime.now(eat_timezone)
    return local_now.strftime('%Y%m%d_%H%M%S')

# Get admin password from environment variable (set in Hugging Face Secrets)
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

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

    /* Expander styling - ENHANCED FOR BETTER VISIBILITY */
    .streamlit-expanderHeader {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #1B5E20 !important;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%) !important;
        border-radius: 16px !important;
        padding: 14px 20px !important;
        margin: 12px 0 !important;
        border: 1px solid #A5D6A7 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    /* Hover effect for expander headers */
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%) !important;
        transform: scale(1.01) !important;
        color: #0D4710 !important;
    }

    /* Make the expander icon (arrow) much larger */
    .streamlit-expanderHeader svg {
        width: 1.6rem !important;
        height: 1.6rem !important;
        margin-right: 12px !important;
        fill: #2E7D32 !important;
    }

    /* Expander content area styling */
    .streamlit-expanderContent {
        padding: 8px 16px !important;
        border-left: 4px solid #2E7D32 !important;
        margin-bottom: 16px !important;
        background-color: #FAFFFA !important;
        border-radius: 0 12px 12px 0 !important;
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

def html_expander(title, content_html, expanded=False, icon="📊"):
    """
    Create an expander using native HTML details/summary tags.
    """
    open_attr = "open" if expanded else ""

    expander_html = f"""
    <details {open_attr} style="
        margin: 12px 0;
        border-radius: 16px;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border: 1px solid #A5D6A7;
        overflow: hidden;
    ">
        <summary style="
            padding: 14px 20px;
            font-size: 1.3rem;
            font-weight: 700;
            color: #1B5E20;
            cursor: pointer;
            list-style: none;
            display: flex;
            align-items: center;
            gap: 12px;
            user-select: none;
        ">
            <span style="font-size: 1.6rem;">{icon}</span>
            <span style="flex: 1;">{title}</span>
            <span style="font-size: 1.4rem;">▼</span>
        </summary>
        <div style="padding: 16px; background-color: #FAFFFA; border-left: 4px solid #2E7D32;">
            {content_html}
        </div>
    </details>
    """

    st.markdown(expander_html, unsafe_allow_html=True)
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

# Batch processing session states
if 'batch_mode' not in st.session_state:
    st.session_state.batch_mode = False
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'show_batch_results' not in st.session_state:
    st.session_state.show_batch_results = False
# Batch options menu states
if 'show_common_chemicals_batch' not in st.session_state:
    st.session_state.show_common_chemicals_batch = False
if 'show_alternative_batch' not in st.session_state:
    st.session_state.show_alternative_batch = None
if 'scroll_to_dropdown' not in st.session_state:
    st.session_state.scroll_to_dropdown = False
if 'highlight_dropdown' not in st.session_state:
    st.session_state.highlight_dropdown = False
# Add to session state initialization section
if 'show_bounding_boxes' not in st.session_state:
    st.session_state.show_bounding_boxes = True
if 'current_raw_heatmap' not in st.session_state:
    st.session_state.current_raw_heatmap = None

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

    # Set Kenya timezone
    kenya_tz = pytz.timezone('Africa/Nairobi')
    local_now = datetime.now(kenya_tz)
    local_time_str = local_now.strftime('%Y-%m-%d %H:%M:%S %Z')

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
    #xai_refs_text += f"  [{grad_cam_ref_num}] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,' in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618-626.\n"

    report = f"""
CROP DOCTOR DIAGNOSIS REPORT
{'='*60}
Date (Local Time): {local_time_str}

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
        1: "SARI, APNI, CSIR-SARI, 'Maize Cropping Guide: 4R Nutrient Management and Best Agronomic Practices, Northern Ghana,' 2022. Available at 'https://www.apni.net/wp-content/uploads/2022/05/4R-Maize-Guide-0511.pdf'",
        2: "SA Grain, 'The big five maize leaf diseases: identification and management,' SA Grain, 2026. Available at https://sagrainmag.co.za/2026/03/05/the-big-five-maize-leaf-diseases-identification-and-management/",
        3: "D. N. Shepherd et al., 'Maize streak virus: an old and complex emerging pathogen,' Molecular Plant Pathology, vol. 11, no. 1, pp. 1-12, 2009. Available at https://bsppjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.1364-3703.2009.00568.x",
        4: "M. K. Haraman, 'Management of maize streak virus disease (MSVD),' CABI Plantwise Knowledge Bank, 2013. Available at https://plantwiseplusknowledgebank.org/doi/10.1079/PWKB.20167800236",
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
        26: "Crop Protection Network, 'Curvularia Leaf Spot of Corn' 2020. Available at 'https://cropprotectionnetwork.org/encyclopedia/curvularia-leaf-spot-of-corn.'",
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
        103: "Gypsoil, 'Gypsum (Calcium Sulphate) Soil Amendment Technical Data Sheet,' 2026.",
        # ============================================================
        # WEATHER AND SPRAYING THRESHOLD REFERENCES (104-111)
        # Added for scientific validation of weather-based recommendations
        # ============================================================

        104: "P. B. Bish, et al., 'Investigating the meteorological effects on drift from a broadcast application of dicamba,' Weed Technology, vol. 37, no. 3, pp. 242-251, 2023. doi: 10.1017/wet.2023.28.",

        105: "Pesticide Safety Directorate, 'Guidance on spraying in relation to weather conditions,' UK Health and Safety Executive, London, UK, 2010. [Online]. Available: https://www.pesticides.gov.uk/guidance/industry/legislation/guidance-on-spraying-in-relation-to-weather-conditions",

        106: "NORAD, 'Pesticide spray drift: A guide for commercial applicators,' Northwest Regional Agricultural Directory (NORAD), 1995. [Online]. Available: https://www.norad.org/research/pesticide-spray-drift-a-guide-for-commercial-applicators/",

        107: "M. M. Dewan, et al., 'Assessing meteorological variables and their impact on pesticide spraying in agricultural areas of Bangladesh,' Research Gate, Jan. 2023. [Online]. Available: https://www.researchgate.net/publication/376773195_Assessing_Meteorological_Variables_and_Their_Impact_on_Pesticide_Spraying_in_Agricultural_Areas_of_Bangladesh",

        108: "R. A. Leonard and J. R. Willian, 'Influence of rainfall intensity and volume on pesticide wash-off from foliage,' Journal of Environmental Science and Health, Part B, vol. 19, no. 6, pp. 521-536, 1984. doi: 10.1080/03601238409372449.",

        109: "A. L. Jones, 'Influence of humidity on fungal disease development in vegetable crops,' Plant Disease, vol. 75, no. 8, pp. 782-789, 1991. doi: 10.1094/PD-75-0782.",

        110: "E. S. Calvo, 'Fenoxaprop-p-ethyl efficacy as a function of temperature and relative humidity,' Planta Daninha, vol. 36, 2018. doi: 10.1590/S0100-83582018360100090.",

        111: "R. R. Granados, et al., 'Survival of plant pathogens and pests under low humidity conditions,' Annual Review of Phytopathology, vol. 59, pp. 239-264, 2021. doi: 10.1146/annurev-phyto-020620-102602."
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

    SCIENTIFIC REFERENCES:
    [1] Pesticide Safety Directorate, "Guidance on spraying in relation to weather conditions,"
        UK Health and Safety Executive, London, UK, 2010.
    [2] E. S. Calvo, "Fenoxaprop-p-ethyl efficacy as a function of temperature and relative humidity,"
        Planta Daninha, vol. 36, 2018. doi: 10.1590/S0100-83582018360100090.
    [3] A. L. Jones, "Influence of humidity on fungal disease development in vegetable crops,"
        Plant Disease, vol. 75, no. 8, pp. 782-789, 1991. doi: 10.1094/PD-75-0782.
    [4] R. R. Granados et al., "Survival of plant pathogens and pests under low humidity conditions,"
        Annual Review of Phytopathology, vol. 59, pp. 239-264, 2021. doi: 10.1146/annurev-phyto-020620-102602.
    [5] R. A. Leonard and J. R. Willian, "Influence of rainfall intensity and volume on pesticide wash-off from foliage,"
        Journal of Environmental Science and Health, Part B, vol. 19, no. 6, pp. 521-536, 1984. doi: 10.1080/03601238409372449.
    [6] P. B. Bish, et al., "Investigating the meteorological effects on drift from a broadcast application of dicamba,"
        Weed Technology, vol. 37, no. 3, pp. 242-251, 2023. doi: 10.1017/wet.2023.28.
    [7] NORAD, "Pesticide spray drift: A guide for commercial applicators,"
        Northwest Regional Agricultural Directory (NORAD), 1995.
    [8] M. M. Dewan, et al., "Assessing meteorological variables and their impact on pesticide spraying in agricultural areas of Bangladesh,"
        Research Gate, Jan. 2023.
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
                    'risk_msg': "🌿 HEALTHY CROP - Continue good farming practices. Regular monitoring recommended.",
                    'risk_class': "risk-low",
                    'location': location,
                    'location_source': location_source,
                    'lat': lat,
                    'lon': lon
                }

            # ============================================================
            # GENERATE WEATHER-BASED RISK ASSESSMENT
            # SCIENTIFIC THRESHOLDS:
            # - Humidity >80%: Fungal disease risk [109]
            # - Wind <3 km/h: Inversion risk [104]
            # - Wind >25 km/h: Spray drift prohibition [106]
            # - Rain 2-5mm: >50% pesticide wash-off [108]
            # ============================================================

            risk_msg = None
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
                is_fungal = 'fungal' in category or any(word in full_text for word in ['fungus', 'fungal', 'spores', 'mycelium', 'blight', 'rust', 'mildew'])
                is_viral = 'viral' in category or any(word in full_text for word in ['virus', 'vector', 'whitefly', 'aphid', 'curl', 'mosaic'])
                is_pest = 'pest' in category or any(word in full_text for word in ['pest', 'larvae', 'insect', 'caterpillar', 'worm', 'borer', 'mite', 'aphid'])
                is_bacterial = 'bacterial' in category or any(word in full_text for word in ['bacteria', 'bacterial', 'speck', 'spot', 'wilt'])

                # Generate disease-specific risk assessment if weather links exist
                if has_humidity_link or has_temp_link or has_rain_link or has_dry_link:
                    # FUNGAL DISEASE RISK - Based on [109]: >80% RH threshold
                    if is_fungal and has_humidity_link and humidity != 'N/A':
                        if humidity > 80:  # [109] - 80% RH threshold for fungal development
                            risk_msg = f"⚠️ HIGH FUNGAL DISEASE RISK - High humidity ({humidity}%) favours fungal growth."
                            risk_class = "risk-high"
                        elif humidity > 65:
                            risk_msg = f"🟡 MODERATE FUNGAL DISEASE RISK - Current humidity ({humidity}%) may favour disease development."
                            risk_class = "risk-moderate"
                        else:
                            risk_msg = f"✅ Low fungal disease risk - Current humidity ({humidity}%) is not favourable for disease spread."
                            risk_class = "risk-low"

                    # VIRAL DISEASE RISK (vector activity)
                    elif is_viral and (has_temp_link or 'vector' in full_text) and temp != 'N/A':
                        if temp > 25 and humidity > 60:
                            risk_msg = f"⚠️ HIGH VIRAL DISEASE RISK - Warm, humid conditions ({temp}°C, {humidity}%) favour vector activity."
                            risk_class = "risk-high"
                        elif temp > 22:
                            risk_msg = f"🟡 MODERATE VIRAL DISEASE RISK - Current conditions ({temp}°C) may support vector activity."
                            risk_class = "risk-moderate"
                        else:
                            risk_msg = f"✅ Low viral disease risk - Current conditions ({temp}°C) are less favourable for vectors."
                            risk_class = "risk-low"

                    # PEST INFESTATION RISK - Based on [106]: dry conditions favor pests
                    elif is_pest and has_dry_link and temp != 'N/A':
                        if temp > 25 and rain_sum < 5:
                            risk_msg = f"⚠️ HIGH PEST RISK - Warm, dry conditions ({temp}°C) favour pest activity."
                            risk_class = "risk-high"
                        elif temp > 20:
                            risk_msg = f"🟡 MODERATE PEST RISK - Current temperature ({temp}°C) may favour pest activity."
                            risk_class = "risk-moderate"
                        else:
                            risk_msg = f"✅ Low pest risk - Current temperature ({temp}°C) is less favourable for pests."
                            risk_class = "risk-low"

                    # RAIN-SPREAD DISEASES - Based on [108]: 2-5mm wash-off threshold
                    elif has_rain_link:
                        if rain_sum > 10 or rain > 1:
                            risk_msg = f"⚠️ HIGH DISEASE SPREAD RISK - Rainfall ({rain_sum}mm) can spread disease."
                            risk_class = "risk-high"
                        elif rain_prob > 50:
                            risk_msg = f"🟡 MODERATE DISEASE SPREAD RISK - Expected rainfall may facilitate spread."
                            risk_class = "risk-moderate"
                        else:
                            risk_msg = f"✅ Low disease spread risk - Dry conditions reduce spread."
                            risk_class = "risk-low"

            # ============================================================
            # If no disease-specific risk message was generated, use GENERAL WEATHER ADVICE
            # Based on thresholds from [104], [105], [106], [108], [109]
            # ============================================================

            if not risk_msg:
                # Build general weather-based advice
                advice_parts = []
                risk_level = "low"

                # Temperature advice - Based on [105]: 10-30°C ideal range
                if temp != 'N/A' and temp is not None:
                    try:
                        temp_val = float(temp)
                        if temp_val > 30:  # [105] - >30°C causes heat stress
                            advice_parts.append(f"🌡️ Hot ({temp_val:.0f}°C): Heat stress possible. Ensure adequate irrigation.")
                            risk_level = "moderate"
                        elif temp_val < 15:  # [105] - <15°C slows growth
                            advice_parts.append(f"🌡️ Cool ({temp_val:.0f}°C): Slow crop growth. Delay sensitive operations.")
                        else:
                            advice_parts.append(f"🌡️ Moderate temperature ({temp_val:.0f}°C) - favorable for crop growth.")
                    except (ValueError, TypeError):
                        pass

                # Humidity advice - Based on [109]: >80% RH threshold for fungal risk
                if humidity != 'N/A' and humidity is not None:
                    try:
                        hum_val = float(humidity)
                        if hum_val > 80:  # [109] - >80% RH favors fungal diseases
                            if treatment_data and 'fungal' in treatment_data.get('category', '').lower():
                                advice_parts.append(f"💧 High humidity ({hum_val:.0f}%): Fungal disease risk increases. Consider preventive spraying.")
                                risk_level = "high" if risk_level != "high" else risk_level
                            else:
                                advice_parts.append(f"💧 High humidity ({hum_val:.0f}%): Monitor for disease development.")
                                risk_level = "moderate" if risk_level != "high" else risk_level
                        elif hum_val < 40:  # [106] - Low humidity favors pests
                            if treatment_data and 'pest' in treatment_data.get('category', '').lower():
                                advice_parts.append(f"💧 Low humidity ({hum_val:.0f}%): Pest risk increases. Check for aphids, mites.")
                                risk_level = "moderate"
                            else:
                                advice_parts.append(f"💧 Low humidity ({hum_val:.0f}%): May increase plant water stress.")
                        else:
                            advice_parts.append(f"💧 Moderate humidity ({hum_val:.0f}%) - good for crop health.")
                    except (ValueError, TypeError):
                        pass

                # Rain advice - Based on [108]: 2-5mm washes off >50% of pesticides
                try:
                    rain_val = float(rain_sum) if rain_sum else 0
                    if rain_val > 10:  # [108] - >10mm causes complete wash-off
                        advice_parts.append(f"☔ Heavy rain expected ({rain_val:.1f}mm): Postpone spraying. Protect young plants.")
                        risk_level = "high"
                    elif rain_val > 5:  # [108] - 5-10mm causes significant wash-off
                        advice_parts.append(f"☔ Moderate rain ({rain_val:.1f}mm): Use rain-fast products if spraying urgent.")
                        risk_level = "moderate" if risk_level != "high" else risk_level
                    elif rain_val > 2.5:  # [108] - 2.5-5mm threshold for wash-off
                        advice_parts.append(f"☔ Light to moderate rain ({rain_val:.1f}mm): Consider delaying application.")
                    elif rain_val > 0:
                        advice_parts.append(f"☔ Light rain ({rain_val:.1f}mm): Safe for most field operations.")
                    elif rain_prob and float(rain_prob) > 70:
                        advice_parts.append(f"☔ High rain chance ({rain_prob:.0f}%): Consider delaying application.")
                except (ValueError, TypeError):
                    pass

                # Wind advice - Based on [104], [106]: <3 km/h inversion risk, >25 km/h drift prohibition
                if wind != 'N/A' and wind is not None:
                    try:
                        wind_val = float(wind)
                        if wind_val > 25:  # [106] - >25 km/h: DO NOT SPRAY
                            advice_parts.append(f"💨 Strong wind ({wind_val:.0f} km/h): High drift risk. DO NOT SPRAY.")
                            risk_level = "high"
                        elif wind_val > 15:  # [106] - 15-25 km/h: caution zone
                            advice_parts.append(f"💨 Moderate wind ({wind_val:.0f} km/h): Drift risk. Use drift-reduction nozzles.")
                            risk_level = "moderate" if risk_level != "high" else risk_level
                        elif wind_val < 3:  # [104] - <3 km/h: inversion risk
                            advice_parts.append(f"💨 Calm wind ({wind_val:.0f} km/h): Check for temperature inversions before spraying (because a layer of cool, dense air traps spray droplets near the ground, causing them to drift off-target and damage neighboring plants). You can test for an inversion either conducting 'Smoke Test' or using 'Temperature Readings. 1. Smoke Test: Ignite a smoke bomb or kick up a small amount of dust. If the smoke gathers into a stationary cloud or hangs flat instead of mixing into the atmosphere, do not spray. 2. Temperature Readings: Measure the temperature at ground level and at about 2 to 3 meters (7 to 10 feet) up. If the upper air is warmer than the surface air, an inversion exists - by Corteva Agriscience.")
                        elif wind_val < 5.8:  # [104] - <5.8 km/h: stable atmospheric conditions
                            advice_parts.append(f"💨 Light wind ({wind_val:.0f} km/h): Verify no inversion present before spraying. Avoid spraying during a temperature inversion because a layer of cool, dense air traps spray droplets near the ground, causing them to drift off-target and damage neighboring plants. You can test for an inversion either conducting 'Smoke Test' or using 'Temperature Readings. 1. Smoke Test: Ignite a smoke bomb or kick up a small amount of dust. If the smoke gathers into a stationary cloud or hangs flat instead of mixing into the atmosphere, do not spray. 2. Temperature Readings: Measure the temperature at ground level and at about 2 to 3 meters (7 to 10 feet) up. If the upper air is warmer than the surface air, an inversion exists - by Corteva Agriscience.")
                        else:
                            advice_parts.append(f"💨 Moderate wind ({wind_val:.0f} km/h) - good for spraying.")
                    except (ValueError, TypeError):
                        pass

                # Add disease-specific note if available
                if treatment_data:
                    category = treatment_data.get('category', '')
                    if 'fungal' in category.lower():
                        advice_parts.append("🍄 **Fungal disease**: Preventive fungicides work best before infection.")
                    elif 'pest' in category.lower():
                        advice_parts.append("🐛 **Pest infestation**: Scout regularly. Early detection allows smaller pesticide amounts.")
                    elif 'viral' in category.lower():
                        advice_parts.append("🦠 **Viral disease**: Control insect vectors. Remove infected plants immediately.")
                    elif 'bacterial' in category.lower():
                        advice_parts.append("🦠 **Bacterial disease**: Copper-based products can help prevent spread.")

                if advice_parts:
                    risk_msg = " | ".join(advice_parts)
                else:
                    risk_msg = "Monitor your crop regularly for any signs of disease development. Good farming practices are your best defense."

                # Set risk class based on highest risk level found
                if risk_level == "high":
                    risk_class = "risk-high"
                elif risk_level == "moderate":
                    risk_class = "risk-moderate"
                else:
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
    """Display help information - Same style as List of Supported Classes with scientific references

    SCIENTIFIC REFERENCES FOR WEATHER THRESHOLDS:
    [1] P. B. Bish et al., "Investigating the meteorological effects on drift," Weed Technology, vol. 37, no. 3, pp. 242-251, 2023.
    [2] Pesticide Safety Directorate, "Guidance on spraying in relation to weather conditions," UK HSE, 2010.
    [3] NORAD, "Pesticide spray drift: A guide for commercial applicators," Northwest Regional Agricultural Directory, 1995.
    [4] M. M. Dewan et al., "Assessing meteorological variables and their impact on pesticide spraying," Research Gate, 2023.
    [5] R. A. Leonard and J. R. Willian, "Influence of rainfall intensity on pesticide wash-off," J Environ Sci Health B, vol. 19, no. 6, pp. 521-536, 1984.
    [6] A. L. Jones, "Influence of humidity on fungal disease development," Plant Disease, vol. 75, no. 8, pp. 782-789, 1991.
    [7] E. S. Calvo, "Fenoxaprop-p-ethyl efficacy as a function of temperature and relative humidity," Planta Daninha, vol. 36, 2018.
    [8] R. R. Granados et al., "Survival of plant pathogens and pests under low humidity conditions," Annual Review of Phytopathology, vol. 59, pp. 239-264, 2021.
    """
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

        **📱 How to Add Crop Doctor to Your Phone Home Screen**

        This creates an icon on your phone that opens the app directly - no need to type the web address each time!

        **For Android (Chrome Browser):**
        • Open Crop Doctor in Chrome
        • Tap the three dots ⋮ in the top right corner
        • Tap "Add to Home Screen"
        • Name it "CropDoc" and tap "Add"
        • The 🌾 icon will appear on your home screen

        **For iPhone/iPad (Safari Browser):**
        • Open Crop Doctor in Safari
        • Tap the Share button (square with arrow pointing up)
        • Scroll down and tap "Add to Home Screen"
        • Name it "Crop Doctor" and tap "Add"
        • The 🌾 icon will appear on your home screen

        **For Desktop (Chrome/Edge Browser):**
        • Click the install icon (monitor with down arrow) in the address bar
        • Or click the three dots ⋮ → "Install Crop Doctor"
        • The app will open in its own window

        ---

        **💡 Tips for Best Results**

        • Take photos of affected leaves showing clear symptoms
        • Include a healthy leaf for comparison if possible
        • Avoid blurry or poorly lit images
        • For low confidence predictions, consult a local expert
        • Always verify chemical recommendations with local agrovets

        ---

        **🌤️ Understanding Weather Information**

        Hover over the ❔ icon next to any weather parameter to get detailed explanations with scientific references [1-8]:

        <table style="width:100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background-color: #2E7D32; color: white;">
                <th style="padding: 8px; border: 1px solid #ddd;">Parameter</th>
                <th style="padding: 8px; border: 1px solid #ddd;">What the tooltip explains</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Scientific Reference</th>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">🌡️ Temperature</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Current air temperature and its effect on crops</td>
                <td style="padding: 8px; border: 1px solid #ddd;">[2], [7]</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">💧 Humidity</td>
                <td style="padding: 8px; border: 1px solid #ddd;">How humidity affects diseases and pests</td>
                <td style="padding: 8px; border: 1px solid #ddd;">[6], [8]</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">☔ Rainfall</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Safe amounts for spraying vs. wash-off risk</td>
                <td style="padding: 8px; border: 1px solid #ddd;">[5]</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">🌬️ Wind Speed</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Best conditions for spraying and drift risk</td>
                <td style="padding: 8px; border: 1px solid #ddd;">[1], [3]</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">🌧️ Rain Probability</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Chance of rain and how much to expect</td>
                <td style="padding: 8px; border: 1px solid #ddd;">[5]</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">🎯 Disease Risk</td>
                <td style="padding: 8px; border: 1px solid #ddd;">What the risk level means for your crops</td>
                <td style="padding: 8px; border: 1px solid #ddd;">[1-8]</td>
            </tr>
        </table>

        **📖 Quick Weather Reference (Scientific Thresholds) [1-8]:**

        <table style="width:100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background-color: #2E7D32; color: white;">
                <th style="padding: 8px; border: 1px solid #ddd;">Condition</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Threshold</th>
                <th style="padding: 8px; border: 1px solid #ddd;">What it means</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Action</th>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">🌡️ Temperature</td>
                <td style="padding: 8px; border: 1px solid #ddd;">10-30°C</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Ideal range for crop growth and pesticide absorption [2]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">✅ Optimal conditions</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">🌡️ High Temperature</td>
                <td style="padding: 8px; border: 1px solid #ddd;">>30°C</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Heat stress, increased evaporation, reduced efficacy [7]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">⚠️ Increase irrigation</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">💧 High Humidity</td>
                <td style="padding: 8px; border: 1px solid #ddd;">>80%</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Fungal disease risk increases [6]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">🍄 Apply preventive fungicide</td>
             </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">💧 Low Humidity</td>
                <td style="padding: 8px; border: 1px solid #ddd;"><40%</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Pest risk (aphids, spider mites) [8]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">🔍 Monitor for pests</td>
             </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">☔ Light Rain</td>
                <td style="padding: 8px; border: 1px solid #ddd;"><2mm</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Safe for spraying, minimal wash-off [5]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">✅ Safe to spray</td>
             </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">☔ Moderate Rain</td>
                <td style="padding: 8px; border: 1px solid #ddd;">2-5mm</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Washes off >50% of pesticides [5]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">⚠️ Use rain-fast products</td>
             </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">☔ Heavy Rain</td>
                <td style="padding: 8px; border: 1px solid #ddd;">>10mm</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Complete wash-off, crop damage risk [5]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">❌ Delay spraying</td>
             </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">🌬️ Calm Wind</td>
                <td style="padding: 8px; border: 1px solid #ddd;"><3 km/h</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Temperature inversion risk [1]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">⚠️ Check for inversions</td>
             </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">🌬️ Ideal Wind</td>
                <td style="padding: 8px; border: 1px solid #ddd;">3-15 km/h</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Minimal drift, good coverage [2]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">✅ Best for spraying</td>
             </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">🌬️ Strong Wind</td>
                <td style="padding: 8px; border: 1px solid #ddd;">>25 km/h</td>
                <td style="padding: 8px; border: 1px solid #ddd;">High drift risk, prohibited [3]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">❌ Do NOT spray</td>
             </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">🌧️ Rain Probability</td>
                <td style="padding: 8px; border: 1px solid #ddd;">>70% with >10mm</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Heavy rain likely [5]</td>
                <td style="padding: 8px; border: 1px solid #ddd;">⚠️ Delay spraying</td>
             </tr>
        </table>

        ---

        **⚠️ Important Notes**

        • This is an AI-assisted diagnostic tool, not a substitute for expert advice
        • Always read product labels before applying any chemicals
        • Follow local regulations and safety guidelines
        • Consult your local agricultural extension officer for confirmation

        ---

        **📚 References (IEEE Style)**

        The weather thresholds used in this app are validated by the following peer-reviewed research
        and authoritative agricultural guidelines:

        **[1]** P. B. Bish, et al., "Investigating the meteorological effects on drift from a broadcast application of dicamba,"
        *Weed Technology*, vol. 37, no. 3, pp. 242-251, 2023. doi: 10.1017/wet.2023.28.

        **[2]** Pesticide Safety Directorate, "Guidance on spraying in relation to weather conditions,"
        *UK Health and Safety Executive*, London, UK, 2010.

        **[3]** NORAD, "Pesticide spray drift: A guide for commercial applicators,"
        *Northwest Regional Agricultural Directory (NORAD)*, 1995.

        **[4]** M. M. Dewan, et al., "Assessing meteorological variables and their impact on pesticide spraying in agricultural areas of Bangladesh,"
        *Research Gate*, Jan. 2023.

        **[5]** R. A. Leonard and J. R. Willian, "Influence of rainfall intensity and volume on pesticide wash-off from foliage,"
        *Journal of Environmental Science and Health, Part B*, vol. 19, no. 6, pp. 521-536, 1984. doi: 10.1080/03601238409372449.

        **[6]** A. L. Jones, "Influence of humidity on fungal disease development in vegetable crops,"
        *Plant Disease*, vol. 75, no. 8, pp. 782-789, 1991. doi: 10.1094/PD-75-0782.

        **[7]** E. S. Calvo, "Fenoxaprop-p-ethyl efficacy as a function of temperature and relative humidity,"
        *Planta Daninha*, vol. 36, 2018. doi: 10.1590/S0100-83582018360100090.

        **[8]** R. R. Granados, et al., "Survival of plant pathogens and pests under low humidity conditions,"
        *Annual Review of Phytopathology*, vol. 59, pp. 239-264, 2021. doi: 10.1146/annurev-phyto-020620-102602.

        ---
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
<h3>🔥 VISUAL EVIDENCE: HOW THE CLASSIFIER MADE THE DECISION </h3>
</div>
""", unsafe_allow_html=True)

    # Display heatmap if available
    if heatmap_overlay is not None and original_img is not None:
        display_heatmap_with_colorbar(original_img, heatmap_overlay, predicted_class, save_path)

    # SECTION 8: GRAD-CAM LEGEND
    st.markdown("""
<div class="section-card">
<h3>📊 LEGEND FOR THE VISUAL OVERLAY COLOURS:</h3>
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

    # ============================================================
    # SECTION 14 REFERENCES (for healthy crops)
    # ============================================================
    management_refs = treatment.get('management_refs', [1])
    references_text = format_reference_list_sequential(management_refs, references)
    references_html = references_text.replace("\n", "<br>")
    html_expander(
            title="VIEW REFERENCES (Click to Expand)",
            content_html=f'<div class="reference-text">{references_html}</div>',
            icon="📚"
        )

    # SECTION 15: WHEN TO CONSULT AN EXPERT (remains outside expander)
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

def display_heatmap_with_colorbar(original_img, heatmap_overlay, predicted_class, save_path=None, show_bounding_boxes=True):
    """Display heatmap with enhanced color bar and optional bounding boxes around high-influence regions"""
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_img, caption="Original Image", width="stretch")

    with col2:
        if show_bounding_boxes:
            # Generate bounding boxes around high-influence regions
            if 'current_raw_heatmap' in st.session_state and st.session_state.current_raw_heatmap is not None:
                raw_heatmap = st.session_state.current_raw_heatmap
                img_with_boxes, boxes, scores = extract_bounding_boxes_from_heatmap(
                    raw_heatmap, original_img, threshold=0.6, min_box_area=500
                )
                st.image(img_with_boxes,
                        caption=f"Visual overlay with bounding boxes (threshold: >60% influence)\n{predicted_class}",
                        width="stretch")

                # Display bounding box statistics
                if boxes:
                    st.caption(f"📍 Found {len(boxes)} high-influence region(s) contributing to this diagnosis. "
                              f"Red boxes indicate >80% influence, orange 70-80%, yellow 60-70% [9].")
                else:
                    st.caption("📍 No high-influence regions above 60% threshold. The model found evidence distributed across the image.")
            else:
                st.image(heatmap_overlay, caption=f"Visual overlay image showing areas that led the model to select:\n{predicted_class}. The redder the area, the higher the confidence of the AI that the area is afflicted by the {predicted_class}.", width="stretch")
        else:
            st.image(heatmap_overlay, caption=f"Visual overlay image showing areas that led the model to select:\n{predicted_class}. The redder the area, the higher the confidence of the AI that the area is afflicted by the {predicted_class}.", width="stretch")

    # Display heatmap with colorbar
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
    ax.annotate('', xy=(1, -1.5), xytext=(0, -1.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, -1.8, 'Influence on Predictions', ha='center', va='top', fontsize=16, fontweight='bold', color='black')

    # Add bounding box threshold indicator
    if show_bounding_boxes:
        ax.axvline(x=0.6, ymin=-2, ymax=0, color='red', linestyle='--', linewidth=2, transform=ax.transAxes)
        ax.text(0.6, -2.2, 'Bounding box threshold (60%)', ha='center', va='top', fontsize=10, color='red', transform=ax.transAxes)

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

    # SECTION 6: MANAGEMENT REFERENCES (in a collapsible expander)
    management_refs_text = format_reference_list_sequential(treatment.get('management_refs', []), references)
    management_refs_html = management_refs_text.replace("\n", "<br>")
    html_expander(
    title="VIEW MANAGEMENT REFERENCES (Click to Expand)",
    content_html=f'<div class="reference-text">{management_refs_html}</div>',
    icon="📚"
    )

    # SECTION 7: CHEMICAL REFERENCES (in a collapsible expander)
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

    chemical_refs_html = chemical_refs_text.replace("\n", "<br>")
    html_expander(
    title="VIEW CHEMICAL REFERENCES (Click to Expand)",
    content_html=f'<div class="reference-text">{chemical_refs_html}</div>',
    icon="📚"
    )

    # SECTION 8: IMPORTANT NOTES (remains outside expander for visibility)
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
    """Display XAI analysis - EACH SECTION in its OWN curved box
    Now includes RED bounding boxes drawn ON the Grad-CAM overlay image
    Matches the layout and styling of batch processing mode

    Scientific reference for grad-cam:
    [9] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra,
        "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,"
        in Proceedings of the IEEE International Conference on Computer Vision (ICCV),
        2017, pp. 618-626.
    """
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

    # Get raw heatmap from disease_data (if available)
    raw_heatmap = disease_data.get('raw_heatmap', None)

    # Also check session state for fallback
    if raw_heatmap is None and 'current_raw_heatmap' in st.session_state:
        raw_heatmap = st.session_state.current_raw_heatmap

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
        title = f"🔬 ALTERNATIVE {alt_num}: EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) ANALYSIS FOR: {predicted_class}"

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

    # ============================================================
    # SECTION 7: VISUAL EVIDENCE
    # ============================================================
    st.markdown("""
<div class="section-card">
<h3>🔥 VISUAL EVIDENCE: HOW THE CLASSIFIER MADE THE DECISION</h3>
</div>
""", unsafe_allow_html=True)

# Initialize boxes to an empty list to avoid UnboundLocalError
    boxes = []
    scores = []

    if heatmap_overlay is not None and original_img is not None:
        show_boxes = st.session_state.get('show_bounding_boxes', True)

        if show_boxes and raw_heatmap is not None:
            overlay_with_boxes, boxes, scores = extract_bounding_boxes_from_heatmap(
                raw_heatmap, heatmap_overlay, threshold=0.6, min_box_area=500
            )

            col_orig, col_overlay = st.columns(2)
            with col_orig:
                st.image(original_img, caption="Original Image", use_container_width=True)
            with col_overlay:
                if boxes and len(boxes) == 1:
                    st.image(overlay_with_boxes,
                            caption=f"The coloured overlay shows how the AI was influenced. The RED box highlights the exact image area that most influenced the diagnosis of {predicted_class}. It represents a region whose area is at least 1% of the image size and the AI was over 60% confident about the diagnosis of {predicted_class}. The percentage inside or above the box shows the AI's confidence that THIS SPECIFIC AREA shows the disease. Higher percentage = stronger evidence.",
                            width="stretch")
                elif boxes and len(boxes) > 1:
                    st.image(overlay_with_boxes,
                            caption=f"The coloured overlay shows how the AI was influenced. The {len(boxes)} RED boxes highlight the exact image areas that most influenced the diagnosis of {predicted_class}. Each RED box represents a region whose area is at least 1% of the image size and the AI was over 60% confident about the diagnosis of {predicted_class}. The percentage inside or above each box shows the AI's confidence that THIS SPECIFIC AREA shows the disease. Higher percentage = stronger evidence.",
                            width="stretch")
                else:
                    st.image(overlay_with_boxes,
                            caption=f"Visual Overlay Image (No RED boxes as there is no single region whose area is at least 1% of the image size where the AI is more than 60% confident about the diagnosis of {predicted_class}.) \nThe disease evidence is spread across the image rather than concentrated in one spot.",
                            use_container_width=True)
        else:
            col_orig, col_overlay = st.columns(2)
            with col_orig:
                st.image(original_img, caption="Original Image", use_container_width=True)
            with col_overlay:
                if not show_boxes:
                    # ALWAYS compute boxes if raw_heatmap is available
                    overlay_with_boxes, boxes, scores = extract_bounding_boxes_from_heatmap(
                        raw_heatmap, heatmap_overlay, threshold=0.6, min_box_area=500
                    )
                    if len(boxes) >0:
                        caption_text = f"Visual overlay image showing graduated influence of areas that led the model to pick on:\n{predicted_class}. The redder the area, the higher the confidence of the AI that the area is afflicted by the {predicted_class}. See the colour bar and legend below. RED bounding boxes are currently HIDDEN."
                    else:
                        caption_text = f"Visual overlay image showing graduated influence of areas that led the model to pick on:\n{predicted_class}. The redder the area, the higher the confidence of the AI that the area is afflicted by the {predicted_class}. See the colour bar and legend below."
                else:
                    caption_text = f"Visual overlay image showing graduated influence of areas that led the model to pick on:\n{predicted_class}. The redder the area, the higher the confidence of the AI that the area is afflicted by the {predicted_class}. See the colour bar and legend below. (Bounding boxes not available for this analysis)"
                st.image(heatmap_overlay, caption=caption_text, use_container_width=True)

        # ============================================================
        # COLOUR BAR
        # ============================================================

        fig, ax = plt.subplots(figsize=(10, 1.2))
        fig.patch.set_visible(False)

        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        spectrum_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        spectrum_cmap = LinearSegmentedColormap.from_list('spectrum', spectrum_colors, N=256)
        ax.imshow(gradient, aspect='auto', cmap=spectrum_cmap, extent=[0, 100, 0, 1])

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(0, -0.4, 'LOW', ha='center', va='top', fontsize=12, fontweight='bold', color='blue')
        ax.text(50, -0.4, 'MEDIUM', ha='center', va='top', fontsize=12, fontweight='bold', color='green')
        ax.text(100, -0.4, 'HIGH', ha='center', va='top', fontsize=12, fontweight='bold', color='red')
        ax.text(50, -0.9, 'Confidence on Predictions', ha='center', va='top', fontsize=13, fontweight='bold', color='black')

        if show_boxes and raw_heatmap is not None:
            ax.axvline(x=60, ymin=0, ymax=1, color='red', linestyle='--', linewidth=2)
            ax.text(60, 1.05, '← 60% (Bounding box threshold)', ha='left', va='bottom',
                   fontsize=9, color='red', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ============================================================
        # TOGGLE BUTTON FOR BOUNDING BOXES
        # ============================================================

        col_toggle1, col_toggle2, col_toggle3 = st.columns([2, 2, 2])
        with col_toggle1:
            pass
        with col_toggle2:
            if show_boxes:
                button_text = "🔴 Hide Bounding Boxes"
                button_help = "Click to remove the red boxes from the overlay image"
            else:
                button_text = "🔴 Show Bounding Boxes"
                button_help = "Click to display red boxes around high-confidence regions (>60%)"

            if st.button(button_text, key="toggle_boxes_xai", width="stretch", help=button_help):
                st.session_state.show_bounding_boxes = not show_boxes
                st.rerun()
        with col_toggle3:
            pass

        if show_boxes:
            st.success("✅ **Bounding boxes are currently VISIBLE.** The red boxes show high-confidence regions (>60%).")
        else:
            st.info("🔲 **Bounding boxes are currently HIDDEN.** Click 'Show Bounding Boxes' to see the high-confidence regions.")

        # ============================================================
        # STATISTICS AND EXPLANATION - WITH ADAPTIVE TEXT
        # ============================================================

        if show_boxes and raw_heatmap is not None and boxes and len(boxes) > 0:
            # Case 1: Bounding boxes found

            # Sort boxes by score (largest to smallest)
            sorted_boxes = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)

            # Build percentage list with box numbers
            percentage_items = []
            for idx, (box, score) in enumerate(sorted_boxes, 1):
                percentage_items.append(f"  Box {idx}: {score*100:.0f}%")
            percentages_text = "<br>".join(percentage_items)

            # Build confidence breakdown text
            high_count = len([s for s in scores if s >= 0.8])
            med_count = len([s for s in scores if 0.7 <= s < 0.8])
            low_count = len([s for s in scores if 0.6 <= s < 0.7])

            breakdown_parts = []
            if high_count > 0:
                breakdown_parts.append(f"🔴 {high_count} region(s) with >80% confidence")
            if med_count > 0:
                breakdown_parts.append(f"🟠 {med_count} region(s) with 70-80% confidence")
            if low_count > 0:
                breakdown_parts.append(f"🟡 {low_count} region(s) with 60-70% confidence")
            breakdown_text = " | ".join(breakdown_parts)

            # Build dynamic example and explanation based on number of boxes
            num_boxes = len(sorted_boxes)

            if num_boxes == 1:
                # SINGLE BOX - Simplified explanation
                stats_content = [
                    f"📍 1 high-confidence region identified.",
                    "",
                    f"The RED box on the overlay image shows the exact image area that most strongly influenced the diagnosis of {predicted_class}.",
                    "",
                    f"📊 **Model Confidence:** {sorted_boxes[0][1]*100:.0f}%",
                    "",
                    f"📈 **Confidence category:** 🔴 >80% confidence.",
                    "",
                    "💡 **What this percentage means:**",
                    f"The AI is {sorted_boxes[0][1]*100:.0f}% confident that the area inside or above the RED box shows signs of {predicted_class}.",
                    "",
                    "✅ **Higher percentages = Stronger evidence** that the disease is present in that specific area."
                ]
            else:
                # MULTIPLE BOXES - Full explanation
                stats_content = [
                    f"📍 {len(boxes)} high-confidence regions identified.",
                    "",
                    f"These RED boxes on the overlay image show the exact image areas that most strongly influenced the diagnosis of {predicted_class}.",
                    "",
                    "📊 **Confidence breakdown (highest to lowest):**",
                    percentages_text,
                    "",
                    f"📈 **Confidence categories:** {breakdown_text}.",
                    "",
                    "💡 **What these percentages mean:**",
                    "Each percentage tells you how confident the AI is that the area inside THAT SPECIFIC BOX shows signs of the disease.",
                    "",
                    "📌 **Important:** These percentages are independent of each other. They do NOT add up to 100%. Each box has its own confidence score.",
                    "",
                    f"🔍 **For example:** The AI is {sorted_boxes[0][1]*100:.0f}% confident the area in Box 1 shows {predicted_class}. It is {sorted_boxes[-1][1]*100:.0f}% confident the area in Box {num_boxes} shows the same disease. Each box is assessed separately.",
                    "",
                    "✅ **Higher percentages = Stronger evidence** that the disease is present in that specific area."
                ]

            stats_html = "<br>".join(stats_content)

            html_expander(
                title="VIEW DETAILED ANALYSIS OF MODEL'S FOCUS AREAS (Click to Expand)",
                content_html=f'<div class="reference-text">{stats_html}</div>',
                icon="📊"
            )

        elif show_boxes and raw_heatmap is not None and confidence >= 0.7:
            # Case 2: No bounding boxes but high confidence - diffuse disease explanation
            stats_content = [
                f"📍 No RED boxes were drawn for this image.",
                "",
                "📦 **Why no boxes?** A RED box is only drawn when there is a contiguous region that is:",
                "   • At least 1% of the total image size, AND",
                "   • Has a confidence score >60%",
                "",
                f"📊 **Global Confidence:** {confidence*100:.1f}%",
                "",
                "🔍 **What this means:**",
                f"{predicted_class} often has diffuse symptoms - the disease evidence is spread across the image rather than concentrated in one large spot. The model recognises the overall pattern of damage across the whole image, even though no single region meets the size threshold for a box.",
                "",
                "💡 Think of it like this: You can recognise a forest from a distance (high global confidence) without being able to point to a single tree that defines it (local bounding boxes).",
                "",
                "📌 **Note:** The coloured overlay still shows the confidence pattern. Warmer colours (red/orange/yellow) indicate areas where the AI had more confidence about the diagnosis, even if they don't meet the size threshold for a box."
            ]

            stats_html = "<br>".join(stats_content)

            html_expander(
                title="VIEW DETAILED ANALYSIS OF MODEL'S FOCUS AREAS (Click to Expand)",
                content_html=f'<div class="reference-text">{stats_html}</div>',
                icon="📊"
            )

        elif not show_boxes:
            stats_content = [
                "🔲 Bounding boxes are currently HIDDEN.",
                "",
                "Click 'Show Bounding Boxes' above to see the red boxes on the overlay image."
            ]

            stats_html = "<br>".join(stats_content)

            html_expander(
                title="VIEW DETAILED ANALYSIS OF MODEL'S FOCUS AREAS (Click to Expand)",
                content_html=f'<div class="reference-text">{stats_html}</div>',
                icon="📊"
            )

        else:
            stats_content = [
                "📍 No high-confidence regions above 60% threshold.",
                "",
                "The disease evidence is distributed across the image rather than concentrated in specific spots. The heatmap colours show the overall confidence pattern."
            ]

            stats_html = "<br>".join(stats_content)

            html_expander(
                title="VIEW DETAILED ANALYSIS OF MODEL'S FOCUS AREAS (Click to Expand)",
                content_html=f'<div class="reference-text">{stats_html}</div>',
                icon="📊"
            )

        if raw_heatmap is None:
            st.info("💡 **Note:** Bounding boxes are not available for this analysis. The standard heatmap overlay is shown instead.")
    else:
        st.warning("⚠️ Visual evidence not available for this diagnosis.")

    # SECTION 8: GRAD-CAM LEGEND
    st.markdown("""
<div class="section-card">
<h3>📊 LEGEND FOR THE VISUAL OVERLAY COLOURS:</h3>
<p>   🔴 <strong>RED (HOT)</strong>     = HIGH confidence (>80%) - These areas strongly indicate the disease</p>
<p>   🟠 <strong>ORANGE</strong>        = HIGH-MEDIUM confidence (70-80%)</p>
<p>   🟡 <strong>YELLOW</strong>        = MEDIUM confidence (60-70%)</p>
<p>   🟢 <strong>GREEN</strong>         = LOW-MEDIUM confidence (40-60%)</p>
<p>   💠 <strong>CYAN</strong>          = VERY LOW confidence (20-40%)</p>
<p>   🔵 <strong>BLUE (COOL)</strong>   = LOWEST confidence (<20%) - These areas had minimal impact on the diagnosis</p>
<p class="gradcam-tip">💡 The warmer the colour (Red → Orange → Yellow), the more confident the AI is that the disease is present. Cooler colours (Green → Cyan → Blue) show areas the AI is less confident about.</p>
<p class="gradcam-tip">📦 <strong>RED BOUNDING BOXES:</strong> Red boxes are drawn directly on the overlay image to highlight regions with >60% confidence.
   The percentage on each red box shows the AI's confidence level for that specific area. These are the specific areas the model relied on the most to make its diagnosis.</p>
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
        references_html = "<br>".join(xai_refs)
        html_expander(
            title="VIEW XAI SOURCES (Click to Expand)",
            content_html=f'<div class="reference-text">{references_html}</div>',
            icon="📚"
        )


# ============================================================
# NEWS AND WEATHER HELPERS FOR ONLINE MODE
# ============================================================

def fetch_kalro_updates(limit=10):
    """Fetch real-time updates from KALRO website"""
    try:
        # KALRO doesn't have an RSS feed, but we can scrape their news section
        response = requests.get("https://kalro.org", timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for news articles on their homepage
            articles = soup.find_all('article', limit=limit)
            # Parse articles...
            pass
    except Exception as e:
        print(f"Error fetching KALRO updates: {e}")

    # Fallback to static sample if scraping fails
    return []

def fetch_kenya_meteo_warnings():
    """Fetch real-time weather warnings from Kenya Meteorological Department"""
    import feedparser

    try:
        # Kenya Met Department CAP RSS feed
        feed = feedparser.parse("https://meteo.go.ke/api/cap/rss.xml")
        warnings = []

        for entry in feed.entries[:5]:
            warnings.append({
                "type": entry.title,
                "areas": entry.get("description", "").split(","),
                "severity": entry.get("severity", "Yellow"),
                "issued": entry.get("published", ""),
                "advice": entry.get("summary", "")
            })
        return warnings
    except Exception as e:
        print(f"Error fetching weather warnings: {e}")
        return []

def fetch_kenya_agriculture_news(query=None, limit=10):
    """Fetch real-time agriculture news from Kenyan news sources"""
    import feedparser
    articles = []

    # Source 1: The Standard - Agriculture RSS Feed (REAL)
    try:
        standard_feed = feedparser.parse("https://www.standardmedia.co.ke/rss/agriculture.php")
        for entry in standard_feed.entries[:limit]:
            articles.append({
                "title": entry.title,
                "summary": entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                "url": entry.link,
                "source": "The Standard",
                "date": entry.get("published", "Recent"),
                "category": "Agriculture"
            })
    except Exception as e:
        print(f"Error fetching The Standard feed: {e}")

    # Source 2: Nation Africa - Agriculture (REAL - if RSS available)
    try:
        # Nation Africa agriculture RSS (verify this URL)
        nation_feed = feedparser.parse("https://nation.africa/kenya/agriculture/rss")
        for entry in nation_feed.entries[:limit]:
            articles.append({
                "title": entry.title,
                "summary": entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                "url": entry.link,
                "source": "Nation Africa",
                "date": entry.get("published", "Recent"),
                "category": "Agriculture"
            })
    except Exception as e:
        print(f"Error fetching Nation Africa feed: {e}")

    # Source 3: Kenya News Agency (KNA) - Agriculture (REAL)
    try:
        # KNA is a government news agency with good agriculture coverage
        kna_feed = feedparser.parse("https://www.kenyanews.go.ke/agriculture/feed/")
        for entry in kna_feed.entries[:limit]:
            articles.append({
                "title": entry.title,
                "summary": entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                "url": entry.link,
                "source": "Kenya News Agency (KNA)",
                "date": entry.get("published", "Recent"),
                "category": "Agriculture"
            })
    except Exception as e:
        print(f"Error fetching KNA feed: {e}")

    return articles

# ============================================================
# REAL-TIME DATA FETCHING FUNCTIONS FOR ONLINE MODE
# ============================================================

def fetch_live_kenya_met_warnings():
    """Fetch live weather warnings from Kenya Meteorological Department RSS feed"""
    import feedparser
    import time

    try:
        # Cache warnings for 1 hour
        cache_key = 'met_warnings_cache'
        cache_time_key = 'met_warnings_time'

        current_time = time.time()
        if cache_key in st.session_state and cache_time_key in st.session_state:
            if current_time - st.session_state[cache_time_key] < 3600:  # 1 hour
                return st.session_state[cache_key]

        feed = feedparser.parse("https://meteo.go.ke/api/cap/rss.xml")
        warnings = []

        for entry in feed.entries[:5]:
            warnings.append({
                "title": entry.get("title", "Weather Alert"),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", "")
            })

        st.session_state[cache_key] = warnings
        st.session_state[cache_time_key] = current_time

        return warnings
    except Exception as e:
        print(f"Error fetching Kenya Met warnings: {e}")
        return []

def fetch_live_agriculture_news():
    """Fetch live agriculture news from reliable Kenyan sources"""
    try:
        # Cache news for 15 minutes
        cache_key = 'news_cache'
        cache_time_key = 'news_time'

        current_time = time.time()
        if cache_key in st.session_state and cache_time_key in st.session_state:
            if current_time - st.session_state[cache_time_key] < 900:
                return st.session_state[cache_key]

        articles = []

        # ============================================================
        # SOURCE 1: The Standard - Agriculture (WORKING)
        # ============================================================
        try:
            standard_feed = feedparser.parse("https://www.standardmedia.co.ke/rss/agriculture.php")
            count = 0
            for entry in standard_feed.entries:
                if count >= 5:
                    break
                articles.append({
                    "title": entry.title,
                    "summary": entry.summary[:300] + "..." if len(entry.summary) > 300 else entry.summary,
                    "url": entry.link,
                    "source": "🌾 The Standard",
                    "date": entry.get("published", "Recent")
                })
                count += 1
            print(f"✅ The Standard: {count} articles")
        except Exception as e:
            print(f"❌ Standard error: {e}")

        # ============================================================
        # SOURCE 2: Kenya News Agency (WORKING)
        # ============================================================
        try:
            kna_feed = feedparser.parse("https://www.kenyanews.go.ke/agriculture/feed/")
            count = 0
            for entry in kna_feed.entries:
                if count >= 5:
                    break
                articles.append({
                    "title": entry.title,
                    "summary": entry.summary[:300] + "..." if len(entry.summary) > 300 else entry.summary,
                    "url": entry.link,
                    "source": "📰 Kenya News Agency",
                    "date": entry.get("published", "Recent")
                })
                count += 1
            print(f"✅ KNA: {count} articles")
        except Exception as e:
            print(f"❌ KNA error: {e}")

        # ============================================================
        # SOURCE 3: Nation Africa - Seeds of Gold (Direct Link Only)
        # Note: No RSS feed available, so we add a direct link
        # ============================================================
        articles.append({
            "title": "Nation Africa - Seeds of Gold",
            "summary": "Visit the Seeds of Gold section for weekly farming pullout, agricultural features, and farming tips from Nation Africa.",
            "url": "https://nation.africa/kenya/business/seeds-of-gold",
            "source": "🌱 Nation Africa",
            "date": "Visit website"
        })

        # ============================================================
        # SOURCE 4: The EastAfrican (RSS with agriculture filter)
        # ============================================================
        try:
            eastafrican_feed = feedparser.parse("https://www.theeastafrican.co.ke/rss")
            count = 0
            for entry in eastafrican_feed.entries:
                if count >= 3:
                    break
                title_lower = entry.title.lower()
                if any(word in title_lower for word in ['farm', 'agriculture', 'crop', 'harvest', 'food']):
                    articles.append({
                        "title": entry.title,
                        "summary": entry.summary[:300] + "..." if len(entry.summary) > 300 else entry.summary,
                        "url": entry.link,
                        "source": "🌍 The EastAfrican",
                        "date": entry.get("published", "Recent")
                    })
                    count += 1
            print(f"✅ EastAfrican: {count} articles")
        except Exception as e:
            print(f"❌ EastAfrican error: {e}")

        # Cache the results
        st.session_state[cache_key] = articles
        st.session_state[cache_time_key] = current_time

        print(f"📊 TOTAL articles: {len(articles)}")
        return articles
    except Exception as e:
        print(f"❌ News fetch error: {e}")
        return []

def fetch_live_kalro_updates():
    """Fetch latest updates from KALRO website"""
    try:
        cache_key = 'kalro_cache'
        cache_time_key = 'kalro_time'

        current_time = time.time()
        if cache_key in st.session_state and cache_time_key in st.session_state:
            if current_time - st.session_state[cache_time_key] < 3600:  # 1 hour
                return st.session_state[cache_key]

        response = requests.get("https://kalro.org", timeout=15)
        updates = []

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('article', limit=3)

            for article in articles:
                title_elem = article.find('h2') or article.find('h3') or article.find('a')
                title = title_elem.get_text(strip=True) if title_elem else "KALRO Update"

                summary_elem = article.find('p')
                summary = summary_elem.get_text(strip=True)[:200] if summary_elem else "Click to read more"

                link_elem = article.find('a')
                link = link_elem.get('href') if link_elem else "https://kalro.org"
                if link and not link.startswith('http'):
                    link = "https://kalro.org" + link

                updates.append({
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "source": "KALRO"
                })

        if not updates:
            updates = [{
                "title": "Visit KALRO Website for Latest Updates",
                "summary": "Check the KALRO website for research findings, new seed varieties, and agricultural advisories.",
                "url": "https://kalro.org",
                "source": "KALRO"
            }]

        st.session_state[cache_key] = updates
        st.session_state[cache_time_key] = current_time

        return updates
    except Exception as e:
        print(f"Error fetching KALRO updates: {e}")
        return [{
            "title": "KALRO Website",
            "summary": "Visit kalro.org for the latest agricultural research and advisory updates.",
            "url": "https://kalro.org",
            "source": "KALRO"
        }]

def fetch_live_weather_forecast(location, disease_name, treatment_data=None):
    """Fetch live weather forecast from Open-Meteo API (already working)"""
    return get_weather_with_risk_assessment(location, disease_name, treatment_data)

def display_online_features(disease_name, crop_type, location, treatment_data=None):
    """Display online features including weather, disease risk assessment, and news

    SCIENTIFIC REFERENCES (IEEE Style) - Listed in order of first appearance:
    [1] Pesticide Safety Directorate, "Guidance on spraying in relation to weather conditions,"
        UK Health and Safety Executive, London, UK, 2010.
        Available: https://www.hse.gov.uk/pesticides/topics/using-pesticides/spray-drift/using-outdoors/controlling-spray-drift.htm

    [2] E. S. Calvo, "Fenoxaprop-p-ethyl efficacy as a function of temperature and relative humidity,"
        Planta Daninha, vol. 36, 2018. doi: https://doi.org/10.1590/S0100-83582018360100090

    [3] A. L. Jones, "Influence of humidity on fungal disease development in vegetable crops,"
        Plant Disease, vol. 75, no. 8, pp. 782-789, 1991. doi: https://doi.org/10.1094/PD-75-0782

    [4] R. R. Granados et al., "Survival of plant pathogens and pests under low humidity conditions,"
        Annual Review of Phytopathology, vol. 59, pp. 239-264, 2021. doi: https://doi.org/10.1146/annurev-phyto-020620-102602

    [5] R. A. Leonard and J. R. Willian, "Influence of rainfall intensity and volume on pesticide wash-off from foliage,"
        Journal of Environmental Science and Health, Part B, vol. 19, no. 6, pp. 521-536, 1984. doi: https://doi.org/10.1080/03601238409372449

    [6] P. B. Bish, et al., "Investigating the meteorological effects on drift from a broadcast application of dicamba,"
        Weed Technology, vol. 37, no. 3, pp. 242-251, 2023. doi: https://doi.org/10.1017/wet.2023.28

    [7] NORAD, "Pesticide spray drift: A guide for commercial applicators,"
        Northwest Regional Agricultural Directory (NORAD), 1995.
        Note: This historical document is not available online. Refer to [1] for equivalent guidance.

    [8] M. M. Dewan, et al., "Assessing meteorological variables and their impact on pesticide spraying in agricultural areas of Bangladesh,"
        Research Gate, Jan. 2023.
        Available: https://www.researchgate.net/publication/376773195
    """

    # Define weather references HTML for the expander with full DOIs and URLs
    weather_refs = [
    '[1] Pesticide Safety Directorate, "Guidance on spraying in relation to weather conditions," UK Health and Safety Executive, London, UK, 2010.',
    '[2] E. S. Calvo, "Fenoxaprop-p-ethyl efficacy as a function of temperature and relative humidity," Planta Daninha, vol. 36, 2018. doi: https://doi.org/10.1590/S0100-83582018360100090',
    '[3] A. L. Jones, "Influence of humidity on fungal disease development in vegetable crops," Plant Disease, vol. 75, no. 8, pp. 782-789, 1991.',
    '[4] R. R. Granados, et al., "Survival of plant pathogens and pests under low humidity conditions," Annual Review of Phytopathology, vol. 59, pp. 239-264, 2021.',
    '[5] R. A. Leonard and J. R. Willian, "Influence of rainfall intensity and volume on pesticide wash-off from foliage," Journal of Environmental Science and Health, Part B, vol. 19, no. 6, pp. 521-536, 1984.',
    '[6] P. B. Bish, et al., "Investigating the meteorological effects on drift from a broadcast application of dicamba," Weed Technology, vol. 37, no. 3, pp. 242-251, 2023.',
    '[7] NORAD, "Pesticide spray drift: A guide for commercial applicators," Northwest Regional Agricultural Directory (NORAD), 1995.',
    '[8] M. M. Dewan, et al., "Assessing meteorological variables and their impact on pesticide spraying in agricultural areas of Bangladesh," Research Gate, Jan. 2023.'
]

    # Join with <br> tags
    weather_refs_html = "<br>".join(weather_refs)

    st.markdown("### 📡 ONLINE MODE - LIVE UPDATES")

    # ============================================================
    # SECTION 1: WEATHER & DISEASE RISK
    # ============================================================
    st.markdown("#### 🌤️ CURRENT WEATHER & DISEASE RISK")

    # Clear tip for both desktop and mobile
    st.info("💡 **Tip:** On computers, hover over the ℹ️ icons. On phones, tap and hold the ℹ️ icons for detailed explanations with scientific references.")

    weather = get_weather_with_risk_assessment(location, disease_name, treatment_data)

    if weather:
        risk_class = weather.get('risk_class', '')
        risk_msg = weather.get('risk_msg', '')

        # Use columns for better mobile layout
        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown("**Current Weather**")

        with col2:
            # [1] UK HSE 2010 - Temperature ideal range and cold effects
            # [2] Calvo 2018 - High temperature effects on pesticide efficacy
            temp_value = weather['temperature']
            st.markdown(f"🌡️ **Temperature:** {temp_value}°C",
                       help=f"Current air temperature. Ideal for most crops is 20-30°C [1]. High temperatures (>30°C) cause heat stress and reduce pesticide efficacy [2]. Low temperatures (<15°C) slow growth [1].")

            # [3] Jones 1991 - High humidity fungal risk
            # [4] Granados et al. 2021 - Low humidity pest risk
            humidity_value = weather['humidity']
            st.markdown(f"💧 **Humidity:** {humidity_value}%",
                       help=f"Relative humidity. High humidity (>80%) favours fungal diseases [3]. Low humidity (<40%) favours pests like spider mites and aphids [4]. Ideal range is 40-70% [1].")

            # [5] Leonard & Willian 1984 - Rainfall wash-off thresholds
            rain_value = weather['rain']
            st.markdown(f"☔ **Current Rainfall:** {rain_value} mm",
                       help=f"Rainfall in the last hour. Research shows 2-5 mm of rain washes off >50% of pesticide deposits [5]. Less than 2mm is safe for spraying. More than 10mm can wash off most chemicals [5].")

            # [6] Bish et al. 2023 - Wind speed and inversions
            # [7] NORAD 1995 - High wind prohibition
            # [8] Dewan et al. 2023 - Meteorological variables overview
            wind_value = weather['wind']
            if wind_value != 'N/A':
                st.markdown(f"🌬️ **Wind Speed:** {wind_value} km/h",
                           help=f"Best for spraying: 3-15 km/h [6]. High winds (>25 km/h) cause spray drift and are prohibited [7]. Calm conditions (<3 km/h) may indicate temperature inversions, where droplets can hang in the air for hours [6]. Wind is a critical meteorological variable affecting spray efficacy [8].")

            # [1] UK HSE 2010 - Forecast planning
            if weather['temp_max'] and weather['temp_min']:
                st.markdown(f"📅 **Today's Forecast:** High {weather['temp_max']}°C / Low {weather['temp_min']}°C",
                           help=f"Expected temperature range for today (from midnight to midnight). Use this to plan activities like transplanting or harvesting [1].")

            # [5] Leonard & Willian 1984 - Rain probability wash-off thresholds
            if weather['rain_prob']:
                st.markdown(f"🌧️ **Rain Probability:** {weather['rain_prob']}% (Expected: {weather['rain_sum']} mm)",
                           help=f"Chance of rain during the remaining hours today. {weather['rain_prob']}% means it may rain. Expected rainfall: {weather['rain_sum']}mm. Safe for spraying if under 2mm. Postpone spraying if expected rain exceeds 10mm [5].")

        # Disease risk assessment
        st.markdown(f"**🎯 DISEASE RISK ASSESSMENT FOR {disease_name}:**")
        st.markdown(f"<span class=\"{risk_class}\">{risk_msg}</span>", unsafe_allow_html=True)
        st.caption("ℹ️ Risk assessment is based on current weather conditions and disease characteristics.",
                  help="High risk means conditions favour disease development. Take preventive action like applying fungicides or improving air circulation.")

    else:
        st.info("🌤️ Unable to fetch weather data. Please check your internet connection.")

    # ============================================================
    # SECTION 2: WEATHER-BASED FARMING TIP
    # ============================================================
    st.markdown("#### 💡 WEATHER-BASED FARMING TIP")

    if weather:
        rain_sum = weather.get('rain_sum', 0)
        rain_prob = weather.get('rain_prob', 0)
        temp = weather.get('temperature', 0)

        if isinstance(temp, str):
            try:
                temp = float(temp)
            except:
                temp = 0

        if rain_sum and rain_sum > 10:
            st.warning("🌧️ **Heavy rain expected!** Postpone spraying. Protect young seedlings from waterlogging. [5]")
        elif rain_prob and rain_prob > 70:
            st.info("🌧️ **Rain likely today.** Consider using rain-fast products if spraying is urgent. [5]")
        elif temp and temp > 30:
            st.warning("🔥 **High temperatures.** Ensure adequate irrigation. Apply mulch to retain moisture. [2]")
        elif temp and temp < 15:
            st.info("❄️ **Cool temperatures.** Delay transplanting sensitive crops. [1]")
        else:
            st.success("🌱 **Optimal conditions.** Good time for spraying, fertilising, and field scouting.")
    else:
        st.info("🌱 Check local weather for optimal farming activities.")

    # ============================================================
    # SECTION 3: SCIENTIFIC REFERENCES (DISPLAYED BEFORE WEATHER WARNINGS)
    # ============================================================

    # Call html_expander exactly like XAI Sources pattern
    html_expander(
        title="VIEW SCIENTIFIC REFERENCES FOR WEATHER THRESHOLDS (Click to Expand)",
        content_html=f'<div class="reference-text">{weather_refs_html}</div>',
        icon="🌤️"
    )

    #st.markdown("---")

    # ============================================================
    # SECTION 4: KENYA MET WEATHER WARNINGS
    # ============================================================
    st.markdown("#### 🚨 KENYA MET WEATHER WARNINGS")
    st.caption("Real-time alerts from Kenya Meteorological Department")

    with st.spinner("🔄 Checking for active weather warnings..."):
        met_warnings = fetch_live_kenya_met_warnings()

    if met_warnings:
        for warning in met_warnings:
            with st.expander(f"⚠️ {warning['title']}"):
                st.caption(f"Published: {warning['published']}")
                st.write(warning['summary'])
                if warning['link']:
                    st.markdown(f"[View official alert]({warning['link']})")
    else:
        st.info("✅ No active weather warnings at this time.")
        st.caption("ℹ️ Follow [@MeteoKenya](https://twitter.com/MeteoKenya) on X for real-time updates")

    # ============================================================
    # SECTION 5: REAL-TIME WEATHER ALERTS
    # ============================================================
    st.markdown("#### 📱 REAL-TIME WEATHER ALERTS")

    st.info("""
    **Follow the Kenya Meteorological Department on X (Twitter) for live updates**

    The Kenya Meteorological Department (`@MeteoKenya`) uses X to issue **immediate** weather warnings, daily forecasts, and heavy rainfall advisories.

    ✅ **Why follow?**
    - Get severe weather alerts instantly (heavy rain, floods, strong winds)
    - Receive daily and 5-day weather forecasts
    - Stay informed about conditions affecting your farm

    👉 **[Follow @MeteoKenya on X](https://twitter.com/MeteoKenya)** (No account needed. Just click to view)
    """)

    # ============================================================
    # SECTION 6: LIVE AGRICULTURE NEWS
    # ============================================================
    st.markdown("#### 📰 LATEST AGRICULTURE NEWS")
    st.caption("Live updates from The Standard and Kenya News Agency")

    with st.spinner("📰 Fetching latest agriculture news... Please wait"):
        news_articles = fetch_live_agriculture_news()

    if news_articles:
        for article in news_articles:
            if article.get('date') == "Visit website" or article.get('source') == "🌱 Nation Africa":
                st.markdown(f"🔗 **{article['source']}** : [{article['title']}]({article['url']})")
                st.caption(article['summary'])
            else:
                with st.expander(f"📰 {article['title']}"):
                    st.caption(f"Source: {article['source']} | {article['date']}")
                    st.write(article['summary'])
                    st.markdown(f"[Read full article]({article['url']})")
    else:
        st.info("📭 No recent news found. Please check your internet connection.")
        st.markdown("""
        **📌 Direct links to agriculture news:**
        - [The Standard - FarmKenya](https://www.standardmedia.co.ke/farmkenya)
        - [Kenya News Agency - Agriculture](https://www.kenyanews.go.ke/agriculture/)
        - [Nation Africa - Seeds of Gold](https://nation.africa/kenya/business/seeds-of-gold)
        """)

    # ============================================================
    # SECTION 7: RESOURCE DIRECTORY
    # ============================================================
    resources_list = [
        '<strong>LIVE NEWS & WEATHER:</strong>',
        '• <a href="https://meteo.go.ke" target="_blank">Kenya Meteorological Department</a> - Official weather warnings',
        '• <a href="https://www.standardmedia.co.ke/farmkenya" target="_blank">The Standard - FarmKenya</a> - Agriculture news',
        '• <a href="https://www.kenyanews.go.ke/agriculture/" target="_blank">Kenya News Agency - Agriculture</a> - Government news',
        '• <a href="https://nation.africa/kenya/business/seeds-of-gold" target="_blank">Nation Africa - Seeds of Gold</a> - Weekly farming pullout',
        '',
        '<strong>RESEARCH & ADVISORY:</strong>',
        '• <a href="https://kalro.org" target="_blank">Kenya Agricultural and Livestock Research Organization (KALRO)</a> - Agricultural research',
        '• <a href="https://www.kephis.org" target="_blank">Kenya Plant Health Inspectorate Service (KEPHIS)</a> - Seed certification',
        '• <a href="https://kilimo.go.ke" target="_blank">Ministry of Agriculture</a> - Government policies',
        '',
        '<strong>FOLLOW ON X (TWITTER) FOR INSTANT UPDATES:</strong>',
        '• <a href="https://twitter.com/MeteoKenya" target="_blank">@MeteoKenya</a> - Weather warnings',
        '• <a href="https://twitter.com/KALROKenya" target="_blank">@KALROKenya</a> - Research updates',
        '• <a href="https://twitter.com/FarmKenya" target="_blank">@FarmKenya</a> - Agriculture news',
        '• <a href="https://twitter.com/SeedsOfGold" target="_blank">@SeedsOfGold</a> - Farming features',
        '',
        '<strong>FARMER SUPPORT:</strong>',
        '• National Agricultural Extension Hotline: <strong>0800 720 123</strong>'
    ]

    # Join with <br> tags
    resources_html = "<br>".join(resources_list)

    # Call html_expander exactly like XAI Sources pattern
    html_expander(
        title="VIEW AGRICULTURAL RESOURCES FOR KENYAN FARMERS (Click to Expand)",
        content_html=f'<div class="reference-text">{resources_html}</div>',
        icon="📚"
    )
def display_weather_references():
    """Display weather threshold references for user transparency

    This function provides full IEEE-style citations for all weather-related
    thresholds used in the Crop Doctor application.
    """
    with st.expander("📚 Scientific References for Weather Thresholds", expanded=False):
        st.markdown("""
        The weather thresholds used in this app are validated by the following peer-reviewed research
        and authoritative agricultural guidelines:

        ---

        **WIND SPEED & TEMPERATURE INVERSIONS:**

        **[104]** P. B. Bish, et al., "Investigating the meteorological effects on drift from a broadcast application of dicamba,"
        *Weed Technology*, vol. 37, no. 3, pp. 242-251, 2023.
        *Key finding: Low wind speed (< 5.8 km/h) is the primary indicator of atmospheric stability and temperature inversions.*

        **[105]** Pesticide Safety Directorate, "Guidance on spraying in relation to weather conditions,"
        *UK Health and Safety Executive*, London, UK, 2010.
        *Key finding: Ideal spraying conditions: 3-15 km/h wind speed, 10-30°C temperature.*

        **[106]** NORAD, "Pesticide spray drift: A guide for commercial applicators,"
        *Northwest Regional Agricultural Directory (NORAD)*, 1995.
        *Key finding: Strong wind (>25 km/h) causes significant pesticide spreading; do not spray.*

        ---

        **RAINFALL & PESTICIDE WASH-OFF:**

        **[108]** R. A. Leonard and J. R. Willian, "Influence of rainfall intensity and volume on pesticide wash-off from foliage,"
        *Journal of Environmental Science and Health, Part B*, vol. 19, no. 6, pp. 521-536, 1984.
        *Key finding: 2-5 mm of rainfall washes off >50% of pesticide deposits within one hour of application.*

        ---

        **HUMIDITY & FUNGAL DISEASE DEVELOPMENT:**

        **[109]** A. L. Jones, "Influence of humidity on fungal disease development in vegetable crops,"
        *Plant Disease*, vol. 75, no. 8, pp. 782-789, 1991.
        *Key finding: Relative humidity >80% is the threshold for active fungal disease development.*

        ---

        **ADDITIONAL RESOURCES:**

        **[107]** M. M. Dewan, et al., "Assessing meteorological variables and their impact on pesticide spraying in agricultural areas of Bangladesh,"
        *Research Gate*, Jan. 2023.
        *Key finding: Comprehensive review of meteorological impacts on pesticide application efficacy.*

        ---
        """)

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

# Integrate Whatsapp
def generate_whatsapp_share_link(diagnosis, confidence, location):
    """Generate a WhatsApp share link with pre-filled message"""

    # Create the message content
    message = f"""🌾 *CROP DOCTOR DIAGNOSIS*

*Disease:* {diagnosis}
*Confidence:* {confidence:.1%}
*Location:* {location}

*Treatment Recommendations:*
See full report for management and chemical control options.

---
Sent via Crop Doctor - AI-Powered Crop Disease Diagnosis and Treatment Recommendation System
https://dosuto-crop-doctor-system.hf.space/
"""

    # URL encode the message
    import urllib.parse
    encoded_message = urllib.parse.quote(message)

    # Create WhatsApp share link
    whatsapp_url = f"https://wa.me/?text={encoded_message}"

    return whatsapp_url

def display_whatsapp_share_button(diagnosis_name, confidence, location):
    """Display a WhatsApp share button for the diagnosis results"""

    # Create the message content with the CORRECT link
    message = f"""🌾 *CROP DOCTOR DIAGNOSIS*

*Disease:* {diagnosis_name}
*Confidence:* {confidence:.1%}
*Location:* {location}

*Next Steps:*
View full treatment recommendations in the Crop Doctor app.

---
Sent via Crop Doctor - AI-Powered Crop Disease Diagnosis and Treatment Recommendation System
https://dosuto-crop-doctor-system.hf.space/
"""

    # URL encode the message
    import urllib.parse
    encoded_message = urllib.parse.quote(message)

    # Create WhatsApp share link
    whatsapp_url = f"https://wa.me/?text={encoded_message}"

    # Display as a button/link - NO UNDERLINE
    st.markdown(f"""
    <a href="{whatsapp_url}" target="_blank" style="text-decoration: none;">
        <div style="
            background: #25D366;
            color: white;
            padding: 8px 12px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 13px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
        ">
            📱 Share via WhatsApp
        </div>
    </a>
    """, unsafe_allow_html=True)

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
        if st.button("📍 Change Location", width="stretch" ):
            st.session_state.show_top_location_dialog = True
            st.rerun()
    with col2:
        if st.button("🔄 Refresh Weather", width="stretch" ):
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
        if st.button("💾 Save Location", width="stretch" ):
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
        if st.button("❌ Cancel", width="stretch" ):
            st.session_state.show_top_location_dialog = False
            st.rerun()

    st.markdown("---")

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
        if st.button("💾 Save Location", width="stretch" , key="top_save_btn"):
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
        if st.button("❌ Cancel", width="stretch" , key="top_cancel_manual_btn"):
            st.session_state.show_top_manual_entry = False
            st.rerun()

    st.markdown("---")

# Get feedback
def display_feedback_section(disease_name, confidence):
    """Display a feedback section with questions farmers can actually answer"""

    #st.markdown("---")
    st.markdown("#### 📝 Help Improve Crop Doctor")
    #st.caption("Your answers help us serve Kenyan farmers better")

    # Question 1: Have you seen this disease before?
    st.markdown("**1. Have you seen this disease on your farm before?**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("❌ Never seen it", width="stretch" , key="seen_never"):
            save_feedback(disease_name, confidence, "seen_before", "never")
            st.success("✅ Thank you! This helps us track new disease outbreaks.")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("⚠️ Seen it a few times", width="stretch" , key="seen_few"):
            save_feedback(disease_name, confidence, "seen_before", "few_times")
            st.success("✅ Thank you for the information!")
            time.sleep(1)
            st.rerun()

    with col3:
        if st.button("🔄 Seen it many times", width="stretch" , key="seen_many"):
            save_feedback(disease_name, confidence, "seen_before", "many_times")
            st.success("✅ Thank you for the information!")
            time.sleep(1)
            st.rerun()

    # Question 2: Are the recommended products available in your area?
    st.markdown("**2. Are the recommended products available in your area?**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Easily available", width="stretch" , key="available_yes"):
            save_feedback(disease_name, confidence, "products_available", "easily")
            st.success("✅ Good to know! Thank you.")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("⚠️ Some are available", width="stretch" , key="available_some"):
            save_feedback(disease_name, confidence, "products_available", "some")
            st.success("✅ Thank you! We'll note this for your region.")
            time.sleep(1)
            st.rerun()

    with col3:
        if st.button("❌ None available", width="stretch" , key="available_no"):
            save_feedback(disease_name, confidence, "products_available", "none")
            st.warning("⚠️ Thank you for letting us know. We'll work on alternative recommendations for your area.")
            time.sleep(1)
            st.rerun()

    # ============================================================
    # QUESTION 3: Grad-CAM Visual Evidence Validation (NEW)
    # ============================================================
    st.markdown("**3. Looking at the coloured overlay on the image, did the system focus on the correct areas?**")
    st.caption("The overlay shows where the AI looked to make its decision. 🔴 Red areas had the most influence, 🔵 Blue areas had the least. If available, the red bounding box(es) show region(s) whose infulence on the diagnosis is >60%.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Yes, focused on the problem areas", width="stretch" , key="gradcam_yes"):
            save_feedback(disease_name, confidence, "gradcam_accuracy", "yes_correct_focus")
            st.success("✅ Thank you! This confirms the AI is looking at the right places.")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("🤔 Partially correct", width="stretch" , key="gradcam_partial"):
            save_feedback(disease_name, confidence, "gradcam_accuracy", "partial_focus")
            st.success("✅ Thank you! This helps us improve the AI's attention.")
            time.sleep(1)
            st.rerun()

    with col3:
        if st.button("❌ No, focused on wrong areas", width="stretch" , key="gradcam_no"):
            save_feedback(disease_name, confidence, "gradcam_accuracy", "wrong_focus")
            st.warning("⚠️ Thank you for letting us know. This helps us retrain the model to focus on the right symptoms.")
            time.sleep(1)
            st.rerun()

    # Optional follow-up for Grad-CAM
    with st.expander("📝 Tell us more about what the overlay image showed (optional)", expanded=False):
        st.markdown("**What did you notice about the coloured overlay image?**")
        st.caption("For example: 'The red areas were on the healthy parts, not the diseased spots' or 'The coloured overlay image correctly highlighted the brown lesions'")

        gradcam_feedback = st.text_area("",
                                      placeholder="Describe what you saw in the coloured overlay image...",
                                      key="gradcam_feedback_area")

        if st.button("📤 Submit coloured overlay image Feedback", width="stretch" ):
            if gradcam_feedback:
                save_feedback(disease_name, confidence, "gradcam_comments", gradcam_feedback, None)
                st.success("✅ Thank you for the detailed feedback!")
                st.balloons()
                time.sleep(2)
                st.rerun()
            else:
                st.warning("Please enter your feedback before submitting.")

    # Question 4: Was this system helpful?
    st.markdown("**4. Was this system helpful?**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Very helpful", width="stretch" , key="helpful_very"):
            save_feedback(disease_name, confidence, "system_helpful", "very_helpful")
            st.success("✅ Thank you! We're glad Crop Doctor could help you.")
            st.balloons()
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("🤔 Somewhat helpful", width="stretch" , key="helpful_somewhat"):
            save_feedback(disease_name, confidence, "system_helpful", "somewhat_helpful")
            st.success("✅ Thank you! Please share what could be improved in the comments.")
            time.sleep(1)
            st.rerun()

    with col3:
        if st.button("❌ Not helpful", width="stretch" , key="helpful_not"):
            save_feedback(disease_name, confidence, "system_helpful", "not_helpful")
            st.warning("⚠️ We're sorry to hear that. Your feedback helps us improve.")
            time.sleep(1)
            st.rerun()

    # Question 5: Would you use this app again?
    st.markdown("**5. Would you use this app again?**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Yes, definitely", width="stretch" , key="use_again_yes"):
            save_feedback(disease_name, confidence, "use_again", "yes")
            st.success("✅ Thank you! We're honored to serve you.")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("🤔 Maybe", width="stretch" , key="use_again_maybe"):
            save_feedback(disease_name, confidence, "use_again", "maybe")
            st.success("✅ Thank you! Is there something we could improve?")
            time.sleep(1)
            st.rerun()

    with col3:
        if st.button("❌ Probably not", width="stretch" , key="use_again_no"):
            save_feedback(disease_name, confidence, "use_again", "no")
            st.warning("⚠️ We're sorry to hear that. Your feedback helps us improve.")
            time.sleep(1)
            st.rerun()

    # Optional: General comments
    with st.expander("📝 Share additional comments (optional)", expanded=False):
        st.markdown("**Do you have any other suggestions to make Crop Doctor better?**")
        st.caption("Examples: Add more crops, include planting advice, add voice support...")

        general_feedback = st.text_area("",
                                      placeholder="Your suggestions help us improve...",
                                      key="general_feedback_area")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Submit Comments", width="stretch" ):
                if general_feedback:
                    save_feedback(disease_name, confidence, "general_comments", general_feedback, None)
                    st.success("✅ Thank you for your suggestions!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                else:
                    st.warning("Please enter your comments before submitting.")
        with col2:
            if st.button("❌ Cancel", width="stretch" ):
                st.rerun()

    st.caption("💡 Your feedback is anonymous and helps us serve Kenyan farmers better.")

def extract_bounding_boxes_from_heatmap(heatmap, overlay_image, threshold=0.6, min_box_area=500):
    """
    Extract bounding boxes from Grad-CAM heatmap and draw RED boxes on the overlay image.
    Labels are placed ABOVE the box when space allows, INSIDE when near top edge.
    Text size has been reduced to prevent overflow into adjacent boxes.

    Parameters:
    - heatmap: numpy array (2D) from Grad-CAM (values 0-1)
    - overlay_image: numpy array (RGB) - the coloured Grad-CAM overlay image
    - threshold: float (0-1) - minimum heatmap value to consider (default 0.6 captures red/orange regions)
    - min_box_area: int - minimum pixel area for a bounding box (filters out tiny regions)

    Returns:
    - overlay_with_boxes: numpy array (RGB) with RED bounding boxes drawn on the overlay
    - boxes: list of (x, y, w, h) tuples for each bounding box
    - scores: list of confidence scores for each box
    """

    # Make a copy of the overlay image to draw boxes on
    overlay_with_boxes = overlay_image.copy()

    # Ensure overlay is uint8 and in RGB format
    if overlay_with_boxes.dtype != np.uint8:
        if overlay_with_boxes.max() <= 1.0:
            overlay_with_boxes = np.uint8(255 * overlay_with_boxes)
        else:
            overlay_with_boxes = np.uint8(overlay_with_boxes)

    # If overlay has 4 channels (RGBA), convert to RGB
    if overlay_with_boxes.shape[2] == 4:
        overlay_with_boxes = cv2.cvtColor(overlay_with_boxes, cv2.COLOR_RGBA2RGB)

    # Convert RGB to BGR for OpenCV drawing operations
    overlay_bgr = cv2.cvtColor(overlay_with_boxes, cv2.COLOR_RGB2BGR)

    # Resize heatmap to match overlay dimensions
    h, w = overlay_bgr.shape[0], overlay_bgr.shape[1]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply threshold to get high-influence regions
    threshold_mask = (heatmap_resized >= threshold).astype(np.uint8) * 255

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and extract bounding boxes with scores
    boxes = []
    scores = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_box_area:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            roi_heatmap = heatmap_resized[y:y+box_h, x:x+box_w]
            score = float(np.max(roi_heatmap)) if roi_heatmap.size > 0 else 0.0
            boxes.append((x, y, box_w, box_h))
            scores.append(score)

    # Draw RED bounding boxes on the BGR image
    red_colour_bgr = (0, 0, 255)

    # Font settings - REDUCED SIZE for better fit
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45   # Reduced from 0.6
    thickness = 1       # Reduced from 2
    padding = 2         # Smaller padding

    for i, (x, y, box_w, box_h) in enumerate(boxes):
        score = scores[i]

        # Draw rectangle with thickness 2
        cv2.rectangle(overlay_bgr, (x, y), (x + box_w, y + box_h), red_colour_bgr, 2)

        # Prepare label
        label = f"{score*100:.0f}%"
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        # Position label above the box or inside if at top edge (original logic)
        if y - 10 > label_size[1]:
            # Place label ABOVE the box
            label_y = y - padding
            # Draw background for label above box
            cv2.rectangle(overlay_bgr,
                         (x, label_y - label_size[1] - padding * 2),
                         (x + label_size[0] + padding * 2, label_y + padding),
                         red_colour_bgr, -1)
            cv2.putText(overlay_bgr, label, (x + padding, label_y - padding),
                       font, font_scale, (255, 255, 255), thickness)
        else:
            # Place label INSIDE the box at the top
            label_y = y + label_size[1] + padding
            # Draw background for label inside box (smaller to fit)
            cv2.rectangle(overlay_bgr,
                         (x + padding, y + padding),
                         (x + label_size[0] + padding * 2, y + label_size[1] + padding * 2),
                         red_colour_bgr, -1)
            cv2.putText(overlay_bgr, label, (x + padding * 2, label_y),
                       font, font_scale, (255, 255, 255), thickness)

    # Convert back from BGR to RGB for Streamlit display
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb, boxes, scores


def overlay_heatmap_with_bounding_boxes(heatmap, original_img, alpha=0.4, threshold=0.6):
    """
    Create an overlay image with both heatmap AND bounding boxes.
    This combines the coloured heatmap overlay with bounding box annotations.

    Parameters:
    - heatmap: numpy array (2D) from Grad-CAM
    - original_img: PIL Image or numpy array (RGB)
    - alpha: float - transparency of heatmap overlay (0-1)
    - threshold: float - threshold for bounding box extraction

    Returns:
    - combined_overlay: numpy array (RGB) with heatmap and bounding boxes
    - boxes: list of bounding boxes
    - scores: list of influence scores
    """
    # First, create the standard heatmap overlay
    if isinstance(original_img, Image.Image):
        img_array = np.array(original_img)
    else:
        img_array = original_img.copy() if isinstance(original_img, np.ndarray) else np.array(original_img)

    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = np.uint8(255 * img_array)
        else:
            img_array = np.uint8(img_array)

    # Resize heatmap
    h, w = img_array.shape[0], img_array.shape[1]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_normalized = np.clip(heatmap_resized, 0, 1)

    # Create coloured heatmap (spectrum: blue -> cyan -> green -> yellow -> orange -> red)
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

    # Convert original to BGR for OpenCV operations
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Overlay heatmap
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    # Convert back to RGB
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Extract and draw bounding boxes on the overlay
    final_overlay, boxes, scores = extract_bounding_boxes_from_heatmap(
        heatmap, img_array, threshold=threshold, min_box_area=500
    )

    return final_overlay, boxes, scores

def save_feedback(disease_name, confidence, question, answer, comment=None):
    """Save feedback to a local file or Hugging Face dataset"""
    import json
    from datetime import datetime
    import pytz

    feedback_file = "farmer_feedback.json"

    # Get current time in Kenya timezone
    kenya_tz = pytz.timezone('Africa/Nairobi')
    timestamp = datetime.now(kenya_tz).strftime('%Y-%m-%d %H:%M:%S')

    # Load existing feedback
    existing_feedback = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r') as f:
                existing_feedback = json.load(f)
        except:
            pass

    # Create feedback entry
    feedback_entry = {
        "timestamp": timestamp,
        "disease": disease_name,
        "confidence": confidence,
        "question": question,
        "answer": answer,
        "comment": comment,
        "session_id": str(uuid4())[:8]
    }

    # Add new feedback
    existing_feedback.append(feedback_entry)

    # Save to file
    with open(feedback_file, 'w') as f:
        json.dump(existing_feedback, f, indent=2)

    # Also upload to Hugging Face dataset
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            feedback_json = json.dumps(feedback_entry, indent=2)
            filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.json"
            api.upload_file(
                path_or_fileobj=io.BytesIO(feedback_json.encode()),
                path_in_repo=f"feedback/{filename}",
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )
        except Exception as e:
            print(f"Failed to upload feedback: {e}")

    return True

def display_feedback_summary():
    """Display a summary of feedback (for admin/developer view)"""
    feedback_file = "farmer_feedback.json"

    if not os.path.exists(feedback_file):
        st.info("No feedback collected yet.")
        return

    try:
        import json
        with open(feedback_file, 'r') as f:
            feedback = json.load(f)

        st.markdown("### 📊 Feedback Summary")
        st.write(f"Total responses: {len(feedback)}")

        # Calculate average rating
        ratings = [f['rating'] for f in feedback if f.get('rating')]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            st.write(f"Average rating: {avg_rating:.1f}/5.0")

        # Show recent feedback
        with st.expander("📋 Recent Feedback", expanded=False):
            for entry in feedback[-10:]:
                st.markdown(f"**{entry['timestamp']}** - {entry['disease']} ({entry['confidence']:.1%})")
                if entry.get('rating_text'):
                    st.write(f"Rating: {entry['rating_text']}")
                if entry.get('feedback'):
                    st.write(f"Feedback: {entry['feedback']}")
                st.markdown("---")
    except Exception as e:
        st.error(f"Error loading feedback: {e}")

# ============================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================

def process_batch_images(uploaded_files, model, class_names, references, gradcam):
    """Process multiple images and return results with local timestamps, treatments, and bounding boxes
    NOW stores Grad-CAM visualisations for ALL top predictions, not just primary.
    """
    results = []
    from datetime import datetime, timedelta, timezone as dt_timezone

    # Get local timezone (EAT - UTC+3)
    eat_timezone = dt_timezone(timedelta(hours=3))

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Open and process image
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Preprocess and predict
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)[0]

            # Get top K predictions
            indices = np.argsort(predictions)[-st.session_state.current_top_k:][::-1]
            top_predictions = []
            for i in indices:
                top_predictions.append({
                    'class': class_names[i],
                    'confidence': float(predictions[i]),
                    'idx': i
                })

            # ============================================================
            # Generate Grad-CAM visualisations for ALL top predictions
            # ============================================================

            # Store overlay and raw heatmap for each prediction
            alt_overlays = {}
            alt_raw_heatmaps = {}

            for alt_idx, pred in enumerate(top_predictions):
                # Generate raw heatmap (values 0-1)
                raw_heatmap = gradcam.generate_heatmap(img_array, pred['idx'])
                # Generate coloured overlay for display
                overlay = gradcam.overlay_heatmap(raw_heatmap, image)

                alt_overlays[alt_idx] = overlay
                alt_raw_heatmaps[alt_idx] = raw_heatmap

            # Get FULL treatment for top prediction (primary)
            treatment = get_full_treatment(top_predictions[0]['class'], references)

            # Get local timestamp for this image
            local_timestamp = datetime.now(eat_timezone).strftime('%Y-%m-%d %H:%M:%S')

            # Save image to Hugging Face dataset
            try:
                save_user_image_for_training(
                    image,
                    top_predictions[0]['class'],
                    top_predictions[0]['confidence'],
                    top_predictions
                )
                print(f"✅ Uploaded {uploaded_file.name} to Hugging Face")
            except Exception as upload_error:
                print(f"⚠️ Failed to upload {uploaded_file.name}: {upload_error}")

            results.append({
                "filename": uploaded_file.name,
                "image": image,
                "top_predictions": top_predictions,
                "primary_diagnosis": top_predictions[0]['class'],
                "primary_confidence": top_predictions[0]['confidence'],
                "treatment": treatment,
                "management": treatment.get('management', 'Information not available'),
                "chemical_control": treatment.get('chemical_control', 'Information not available'),
                "category": treatment.get('category', 'General'),
                "causal_agent": treatment.get('causal_agent', 'Information not available'),
                "is_healthy": treatment.get('is_healthy', False),
                "timestamp": local_timestamp,
                "status": "success",
                # Store visualisations for ALL alternatives
                "alt_overlays": alt_overlays,          # Dict: {0: overlay, 1: overlay, 2: overlay}
                "alt_raw_heatmaps": alt_raw_heatmaps,  # Dict: {0: raw_heatmap, 1: raw_heatmap, 2: raw_heatmap}
                # For backward compatibility
                "heatmap_overlay": alt_overlays.get(0),
                "raw_heatmap": alt_raw_heatmaps.get(0)
            })
        except Exception as e:
            local_timestamp = datetime.now(eat_timezone).strftime('%Y-%m-%d %H:%M:%S')
            results.append({
                "filename": uploaded_file.name,
                "status": "error",
                "error_message": str(e),
                "timestamp": local_timestamp
            })

    return results


def export_all_images_analysis(results):
    """Export a comprehensive analysis of all images in the batch"""
    from datetime import datetime, timedelta, timezone as dt_timezone

    eat_timezone = dt_timezone(timedelta(hours=3))
    timestamp = datetime.now(eat_timezone).strftime('%Y-%m-%d %H:%M:%S')

    report = []
    report.append("=" * 80)
    report.append("CROP DOCTOR - COMPLETE BATCH ANALYSIS")
    report.append("=" * 80)
    report.append(f"Report Generated: {timestamp}")
    report.append(f"Total Images: {len(results)}")
    report.append("")

    for idx, r in enumerate(results, 1):
        if r['status'] == 'success':
            confidence_value = float(r['primary_confidence']) if hasattr(r['primary_confidence'], 'item') else r['primary_confidence']

            report.append(f"\n{'=' * 80}")
            report.append(f"IMAGE {idx}: {r['filename']}")
            report.append(f"{'=' * 80}")
            report.append(f"Diagnosis: {r['primary_diagnosis']}")
            report.append(f"Confidence: {confidence_value*100:.1f}%")
            report.append(f"Category: {r['category']}")
            report.append(f"Causal Agent: {r['causal_agent']}")
            report.append("")

            if not r.get('is_healthy', False):
                report.append("MANAGEMENT:")
                report.append("-" * 40)
                report.append(r['management'])
                report.append("")
                report.append("CHEMICAL CONTROL:")
                report.append("-" * 40)
                report.append(r['chemical_control'])
            else:
                report.append("STATUS: HEALTHY CROP - No treatment needed")
            report.append("")

    report_text = "\n".join(report)

    txt_filename = f"batch_complete_analysis_{datetime.now(eat_timezone).strftime('%Y%m%d_%H%M%S')}.txt"

    st.download_button(
        label="\U0001F4E5 Download Complete Analysis",
        data=report_text,
        file_name=txt_filename,
        mime="text/plain"
    )

def generate_comprehensive_batch_report(results, batch_timestamp):
    """Generate a single comprehensive report for all images in the batch including all references"""
    from datetime import datetime, timedelta, timezone as dt_timezone

    eat_timezone = dt_timezone(timedelta(hours=3))
    timestamp = datetime.now(eat_timezone).strftime('%Y-%m-%d %H:%M:%S')

    # Helper function to clean category
    def clean_category(category_text):
        category_text = category_text.replace('🍄', '').replace('🦠', '').replace('🐛', '').replace('🌿', '').replace('🌱', '').strip()
        return category_text.capitalize()

    # Helper function to format references
    def format_references(ref_numbers, ref_dict):
        if not ref_numbers:
            return "  See local agricultural extension office for region-specific advice"
        ref_list = []
        for idx, ref_num in enumerate(ref_numbers, 1):
            if ref_num in ref_dict:
                ref_list.append(f"  [{idx}] {ref_dict[ref_num]}")
            else:
                ref_list.append(f"  [{idx}] Reference {ref_num}")
        return "\n".join(ref_list) if ref_list else "  See local agricultural extension office for region-specific advice"

    # Separate healthy and diseased crops
    healthy_crops = [r for r in results if r.get('is_healthy', False)]
    diseased_crops = [r for r in results if not r.get('is_healthy', False)]

    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("CROP DOCTOR - COMPREHENSIVE BATCH ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Report Generated: {timestamp}")
    report_lines.append(f"Batch Processed: {batch_timestamp}")
    report_lines.append(f"Time Zone: East Africa Time (EAT, UTC+3)")
    report_lines.append("")
    report_lines.append(f"Total Images Processed: {len(results)}")
    report_lines.append(f"  - Diseased Crops: {len(diseased_crops)}")
    report_lines.append(f"  - Healthy Crops: {len(healthy_crops)}")
    report_lines.append("")

    # Disease Prevalence Summary
    if diseased_crops:
        report_lines.append("-" * 80)
        report_lines.append("DISEASE PREVALENCE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append("")

        disease_counts = {}
        for r in diseased_crops:
            disease_name = r['primary_diagnosis']
            disease_counts[disease_name] = disease_counts.get(disease_name, 0) + 1

        for disease, count in sorted(disease_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(results)) * 100
            report_lines.append(f"  {disease}: {count} image(s) ({percentage:.1f}%)")
        report_lines.append("")

    # Category Breakdown
    categories = {}
    for r in results:
        cat = clean_category(r['category'])
        categories[cat] = categories.get(cat, 0) + 1

    report_lines.append("-" * 80)
    report_lines.append("CATEGORY BREAKDOWN")
    report_lines.append("-" * 80)
    report_lines.append("")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {cat}: {count}")
    report_lines.append("")

    # Confidence Statistics
    if diseased_crops:
        confidences = []
        for r in diseased_crops:
            conf_val = float(r['primary_confidence']) if hasattr(r['primary_confidence'], 'item') else r['primary_confidence']
            confidences.append(conf_val)

        if confidences:
            report_lines.append("-" * 80)
            report_lines.append("CONFIDENCE STATISTICS (Diseased Crops Only)")
            report_lines.append("-" * 80)
            report_lines.append("")
            report_lines.append(f"  Average Confidence: {sum(confidences)/len(confidences)*100:.1f}%")
            report_lines.append(f"  Highest Confidence: {max(confidences)*100:.1f}%")
            report_lines.append(f"  Lowest Confidence: {min(confidences)*100:.1f}%")
            report_lines.append("")

    # Detailed Results for Each Image with Full References
    report_lines.append("=" * 80)
    report_lines.append("DETAILED RESULTS BY IMAGE")
    report_lines.append("=" * 80)
    report_lines.append("")

    for idx, r in enumerate(results, 1):
        conf_val = float(r['primary_confidence']) if hasattr(r['primary_confidence'], 'item') else r['primary_confidence']

        report_lines.append(f"{'=' * 80}")
        report_lines.append(f"IMAGE #{idx}: {r['filename']}")
        report_lines.append(f"{'=' * 80}")
        report_lines.append("")
        report_lines.append("DIAGNOSIS SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"  Primary Diagnosis: {r['primary_diagnosis']}")
        report_lines.append(f"  Confidence: {conf_val*100:.1f}%")
        report_lines.append(f"  Category: {clean_category(r['category'])}")
        report_lines.append(f"  Causal Agent: {r['causal_agent']}")
        report_lines.append(f"  Is Healthy: {'Yes' if r.get('is_healthy', False) else 'No'}")
        report_lines.append(f"  Processed: {r.get('timestamp', batch_timestamp)}")
        report_lines.append("")

        # Top predictions
        report_lines.append("TOP PREDICTIONS")
        report_lines.append("-" * 40)
        for i, pred in enumerate(r['top_predictions'], 1):
            pred_conf = float(pred['confidence']) if hasattr(pred['confidence'], 'item') else pred['confidence']
            report_lines.append(f"  {i}. {pred['class']}: {pred_conf*100:.1f}%")
        report_lines.append("")

        # Treatment recommendations for diseased crops
        if not r.get('is_healthy', False):
            report_lines.append("TREATMENT RECOMMENDATIONS")
            report_lines.append("-" * 40)
            report_lines.append("")
            report_lines.append("MANAGEMENT PRACTICES:")
            report_lines.append("")
            for line in r['management'].split('\n'):
                if line.strip():
                    report_lines.append(f"  {line}")
            report_lines.append("")
            report_lines.append("CHEMICAL CONTROL:")
            report_lines.append("")
            for line in r['chemical_control'].split('\n'):
                if line.strip():
                    report_lines.append(f"  {line}")
            report_lines.append("")

            # MANAGEMENT REFERENCES
            management_refs = r['treatment'].get('management_refs', [])
            report_lines.append("MANAGEMENT REFERENCES:")
            report_lines.append("-" * 40)
            report_lines.append(format_references(management_refs, get_references()))
            report_lines.append("")

            # CHEMICAL REFERENCES
            chemical_refs = r['treatment'].get('chemical_refs_original', [])
            report_lines.append("CHEMICAL REFERENCES:")
            report_lines.append("-" * 40)
            report_lines.append(format_references(chemical_refs, get_references()))
            report_lines.append("")

            # XAI SOURCES
            xai_refs = r['treatment'].get('xai_ref_numbers', [])
            report_lines.append("XAI SOURCES:")
            report_lines.append("-" * 40)
            report_lines.append(format_references(xai_refs, get_references()))
            report_lines.append("")

            # Grad-CAM Reference
            report_lines.append("XAI SOURCES (Grad-CAM):")
            report_lines.append("-" * 40)
            #report_lines.append("  [1] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,' in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618-626.")
            report_lines.append("")
        else:
            report_lines.append("STATUS: HEALTHY CROP")
            report_lines.append("-" * 40)
            report_lines.append("  No treatment needed. Continue good farming practices.")
            report_lines.append("")

        report_lines.append("")

    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Report generated by Crop Doctor - AI-Powered Crop Disease Diagnosis")
    report_lines.append("For questions or support, contact your local agricultural extension officer")

    # Combine and create download
    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Comprehensive Report",
        data=report_text,
        file_name=f"crop_doctor_batch_report_{datetime.now(eat_timezone).strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        key="download_comprehensive_report"
    )

def display_batch_results(results):
    """Display batch processing results in an organised way with selectable image for detailed analysis
    Now includes bounding boxes for Grad-CAM visualisations and explanations for diffuse diseases
    CORRECTED: Resets alternative selection when a different image is selected.
    """
    from datetime import datetime, timedelta, timezone as dt_timezone
    import csv
    import os

    # Get local timezone (EAT - UTC+3)
    eat_timezone = dt_timezone(timedelta(hours=3))

    st.markdown("### \U0001F4CA Batch Processing Results")
    st.markdown(f"**Total images processed:** {len(results)}")

    # Summary statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']

    st.markdown(f"\u2705 Successful: {len(successful)}")
    st.markdown(f"\u274C Failed: {len(failed)}")

    # Display timestamp of batch processing
    batch_timestamp = datetime.now(eat_timezone).strftime('%Y-%m-%d %H:%M:%S')
    st.caption(f"\U0001F550 Batch processed at: {batch_timestamp} (East Africa Time)")

    # Helper function to clean category
    def clean_category(category_text):
        category_text = category_text.replace('🍄', '').replace('🦠', '').replace('🐛', '').replace('🌿', '').replace('🌱', '').strip()
        return category_text.capitalize()

    if not successful:
        st.warning("No successful image processing results to display.")
        if failed:
            st.markdown("#### \u274C Failed Images")
            for r in failed:
                st.warning(f"**{r['filename']}:** {r['error_message']}")
        return

    # Create summary table
    st.markdown("#### \U0001F4CB Diagnosis Summary Table")
    summary_data = []
    for r in successful:
        confidence_value = float(r['primary_confidence']) if hasattr(r['primary_confidence'], 'item') else r['primary_confidence']
        summary_data.append({
            "Image": r['filename'],
            "Diagnosis": r['primary_diagnosis'],
            "Confidence": f"{confidence_value*100:.1f}%",
            "Category": clean_category(r['category']),
        })
    st.dataframe(summary_data, width="stretch")

    # Export buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\U0001F4CA Export Summary CSV", width="stretch"):
            csv_data = []
            for r in successful:
                conf_val = float(r['primary_confidence']) if hasattr(r['primary_confidence'], 'item') else r['primary_confidence']
                csv_data.append({
                    "filename": r['filename'],
                    "diagnosis": r['primary_diagnosis'],
                    "confidence": conf_val,
                    "confidence_percent": f"{conf_val*100:.1f}%",
                    "category": clean_category(r['category']),
                    "causal_agent": r['causal_agent'],
                    "timestamp": r.get('timestamp', batch_timestamp)
                })

            csv_filename = f"batch_results_{datetime.now(eat_timezone).strftime('%Y%m%d_%H%M%S')}.csv"
            with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "diagnosis", "confidence", "confidence_percent", "category", "causal_agent", "timestamp"])
                writer.writeheader()
                writer.writerows(csv_data)

            with open(csv_filename, 'rb') as f:
                st.download_button("📥 Download CSV", data=f, file_name=csv_filename, mime="text/csv", key="download_csv_batch")
            try:
                os.remove(csv_filename)
            except:
                pass

    with col2:
        if st.button("\U0001F4D1 Export Comprehensive Report", width="stretch"):
            generate_comprehensive_batch_report(successful, batch_timestamp)

    st.markdown("---")

    # ============================================================
    # SELECT IMAGE FOR DETAILED ANALYSIS
    # ============================================================

    # Add a prominent header with icon
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 12px 16px;
        border-radius: 16px;
        margin-bottom: 16px;
        border-left: 5px solid #2E7D32;
    ">
        <h3 style="
            color: #1B5E20;
            margin: 0;
            font-size: 1.2rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <span>🔽</span> SELECT IMAGE FOR DETAILED ANALYSIS
        </h3>
        <p style="
            margin: 5px 0 0 0;
            color: #2E7D32;
            font-size: 0.85rem;
        ">Choose an image to view full diagnosis, treatment recommendations, and visual analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Store the previously selected image index to detect changes
    if 'batch_previous_selected_index' not in st.session_state:
        st.session_state.batch_previous_selected_index = None

    image_options = [f"{r['filename']} - {r['primary_diagnosis']}" for r in successful]
    selected_index = st.selectbox(
        "Select Image",
        options=range(len(image_options)),
        format_func=lambda x: image_options[x],
        key="batch_selected_index",
        label_visibility="collapsed"
    )

    # Check if the selected image has changed
    if st.session_state.batch_previous_selected_index != selected_index:
        st.session_state.show_alternative_batch = None
        st.session_state.batch_previous_selected_index = selected_index

    selected_result = successful[selected_index]

    # ============================================================
    # DETERMINE WHETHER TO SHOW PRIMARY OR ALTERNATIVE
    # ============================================================

    selected_alternative = st.session_state.get('show_alternative_batch', None)

    if selected_alternative is not None and selected_alternative < len(selected_result['top_predictions']):
        display_prediction = selected_result['top_predictions'][selected_alternative]
        is_alternative = True
        alt_index = selected_alternative
    else:
        display_prediction = selected_result['top_predictions'][0]
        is_alternative = False
        alt_index = None

    confidence_value = float(display_prediction['confidence']) if hasattr(display_prediction['confidence'], 'item') else display_prediction['confidence']
    treatment = get_full_treatment(display_prediction['class'], get_references())

    # ============================================================
    # GET THE CORRECT OVERLAY FOR THE SELECTED DIAGNOSIS
    # ============================================================

    if is_alternative and alt_index is not None:
        alt_overlays = selected_result.get('alt_overlays', {})
        alt_raw_heatmaps = selected_result.get('alt_raw_heatmaps', {})
        heatmap_overlay = alt_overlays.get(alt_index)
        raw_heatmap = alt_raw_heatmaps.get(alt_index)
    else:
        heatmap_overlay = selected_result.get('heatmap_overlay')
        raw_heatmap = selected_result.get('raw_heatmap')

    if heatmap_overlay is None:
        heatmap_overlay = selected_result.get('heatmap_overlay')
    if raw_heatmap is None:
        raw_heatmap = selected_result.get('raw_heatmap')

    # ============================================================
    # 1. TOP PREDICTIONS
    # ============================================================
    st.markdown(f"""
<div class="section-card">
<h3>📈 TOP {len(selected_result['top_predictions'])} PREDICTIONS</h3>
<div class="prediction-list">
""", unsafe_allow_html=True)

    for i, pred in enumerate(selected_result['top_predictions']):
        pred_conf = float(pred['confidence']) if hasattr(pred['confidence'], 'item') else pred['confidence']
        if i == 0:
            st.markdown(f'<div class="prediction-row"><span class="prediction-marker">✓.</span> <span class="prediction-name">{pred["class"]}</span>: <span class="prediction-value">{pred_conf*100:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-row"><span class="prediction-marker">{i+1}.</span> <span class="prediction-name">{pred["class"]}</span>: <span class="prediction-value">{pred_conf*100:.1f}%</span></div>', unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # ============================================================
    # 2. XAI ANALYSIS WITH BOUNDING BOXES
    # ============================================================

    predicted_class = display_prediction['class']
    original_img = selected_result['image']
    show_boxes = st.session_state.get('show_bounding_boxes', True)

    # Confidence level determination
    if confidence_value >= 0.9:
        level = 'VERY HIGH'
        level_icon = "🟢"
        level_desc = "Extremely reliable - the disease indicators are very clear and distinct."
    elif confidence_value >= 0.8:
        level = 'HIGH'
        level_icon = "🟢"
        level_desc = "Very reliable - the disease indicators are clearly visible."
    elif confidence_value >= 0.7:
        level = 'HIGH'
        level_icon = "🟡"
        level_desc = "Reliable - the disease indicators are present but not extremely strong."
    elif confidence_value >= 0.6:
        level = 'MODERATE'
        level_icon = "🟡"
        level_desc = "Moderately reliable - some disease indicators are visible."
    elif confidence_value >= 0.5:
        level = 'MODERATE'
        level_icon = "🟠"
        level_desc = "Moderately reliable - disease indicators are present but weak."
    elif confidence_value >= 0.4:
        level = 'LOW'
        level_icon = "🟠"
        level_desc = "Low reliability - disease indicators are unclear or mixed."
    elif confidence_value >= 0.3:
        level = 'LOW'
        level_icon = "🔴"
        level_desc = "Low reliability - weak disease indicators present."
    else:
        level = 'VERY LOW'
        level_icon = "🔴"
        level_desc = "Very low reliability - disease indicators are absent or unclear."

    # XAI Analysis Header
    if is_alternative:
        st.markdown(f"""
<div class="section-card">
<h3>🔬 ALTERNATIVE {alt_index + 1}: EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) ANALYSIS FOR: {predicted_class}</h3>
<p><strong>📊 CLASSIFIER CONFIDENCE SCORE:</strong> {confidence_value*100:.1f}% ({level}) {level_icon}</p>
<p>{level_desc}</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="section-card">
<h3>🔬 EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) ANALYSIS FOR: {predicted_class}</h3>
<p><strong>📊 CLASSIFIER CONFIDENCE SCORE:</strong> {confidence_value*100:.1f}% ({level}) {level_icon}</p>
<p>{level_desc}</p>
</div>
""", unsafe_allow_html=True)

    # Confidence Key
    st.markdown("""
<div class="section-card">
<h3>📊 CONFIDENCE KEY - How to Understand the Score</h3>
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

    # Disease Type
    st.markdown(f"""
<div class="section-card">
<h3>🏷️ DISEASE TYPE</h3>
<p>{treatment['category']}</p>
</div>
""", unsafe_allow_html=True)

    # Causal Agent
    st.markdown(f"""
<div class="section-card">
<h3>🦠 CAUSAL AGENT</h3>
<p>{treatment['causal_agent']}</p>
</div>
""", unsafe_allow_html=True)

    # Key Characteristics
    st.markdown(f"""
<div class="section-card">
<h3>📋 KEY CHARACTERISTICS</h3>
<p style="white-space: pre-line;">{treatment.get('causes_characteristics', 'See detailed description')}</p>
</div>
""", unsafe_allow_html=True)

    # What the Model Analysed
    st.markdown(f"""
<div class="section-card">
<h3>💡 WHAT THE MODEL ANALYSED</h3>
<p>The model examined your crop image and identified visual patterns that match the characteristic appearance of {predicted_class}.</p>
</div>
""", unsafe_allow_html=True)

    # ============================================================
    # VISUAL EVIDENCE SECTION
    # ============================================================
    st.markdown("""
<div class="section-card">
<h3>🔥 VISUAL EVIDENCE: HOW THE CLASSIFIER MADE THE DECISION</h3>
</div>
""", unsafe_allow_html=True)

    # Initialize variables
    boxes = []
    scores = []
    overlay_with_boxes = None

    if heatmap_overlay is not None and original_img is not None:
        # ALWAYS compute boxes if raw_heatmap is available
        if raw_heatmap is not None:
            overlay_with_boxes, boxes, scores = extract_bounding_boxes_from_heatmap(
                raw_heatmap, heatmap_overlay, threshold=0.6, min_box_area=500
            )

        if show_boxes and raw_heatmap is not None:
            # Use the computed overlay_with_boxes with boxes visible
            col_orig, col_overlay = st.columns(2)
            with col_orig:
                st.image(original_img, caption="Original Image", use_container_width=True)
            with col_overlay:
                if boxes and len(boxes) == 1:
                    st.image(overlay_with_boxes,
                            caption=f"The coloured overlay shows how the AI was influenced. The RED box highlights the exact image area that most influenced the diagnosis of {predicted_class}. It represents a region whose area is at least 1% of the image size and the AI was over 60% confident about the diagnosis of {predicted_class}. The percentage inside or above the box shows the AI's confidence that THIS SPECIFIC AREA shows the disease. Higher percentage = stronger evidence.",
                            width="stretch")
                elif boxes and len(boxes) > 1:
                    st.image(overlay_with_boxes,
                            caption=f"The coloured overlay shows how the AI was influenced. The {len(boxes)} RED boxes highlight the exact image areas that most influenced the diagnosis of {predicted_class}. Each RED box represents a region whose area is at least 1% of the image size and the AI was over 60% confident about the diagnosis of {predicted_class}. The percentage inside or above each box shows the AI's confidence that THIS SPECIFIC AREA shows the disease. Higher percentage = stronger evidence.",
                            width="stretch")
                else:
                    st.image(overlay_with_boxes,
                            caption=f"Visual Overlay Image (No RED boxes as there is no single region whose area is more than 1% of the image size and where AI has more than 60% confidence about the diagnosis.) \nThe disease evidence is spread across the image rather than concentrated in one spot.",
                            use_container_width=True)
        else:
            # Boxes are hidden OR raw_heatmap not available
            col_orig, col_overlay = st.columns(2)
            with col_orig:
                st.image(original_img, caption="Original Image", use_container_width=True)
            with col_overlay:
                if not show_boxes:
                    if boxes and len(boxes) > 0:
                        # Boxes EXIST but are HIDDEN
                        caption_text = f"Visual overlay image showing graduated influence of areas that led the model to pick on:\n{predicted_class}. The redder the area, the higher the confidence of the AI that that area is afflicted by the {predicted_class}. See the colour bar and legend below. RED bounding boxes are currently HIDDEN. Click 'Show Bounding Boxes' to see them."
                    else:
                        # No boxes exist (diffuse disease)
                        caption_text = f"Visual overlay image showing graduated influence of areas that led the model to pick on:\n{predicted_class}. The redder the area, the higher the confidence of the AI that that area is afflicted by the {predicted_class}. See the colour bar and legend below. (No RED boxes found for this image)"
                else:
                    caption_text = f"Visual overlay image showing graduated influence of areas that led the model to pick on:\n{predicted_class}. The redder the area, the higher the confidence of the AI that that area is afflicted by the {predicted_class}. See the colour bar and legend below. (Bounding boxes not available for this analysis)"
                st.image(heatmap_overlay, caption=caption_text, use_container_width=True)

        # ============================================================
        # COLOUR BAR
        # ============================================================

        fig, ax = plt.subplots(figsize=(10, 1.2))
        fig.patch.set_visible(False)

        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        spectrum_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        spectrum_cmap = LinearSegmentedColormap.from_list('spectrum', spectrum_colors, N=256)
        ax.imshow(gradient, aspect='auto', cmap=spectrum_cmap, extent=[0, 100, 0, 1])

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(0, -0.4, 'LOW', ha='center', va='top', fontsize=12, fontweight='bold', color='blue')
        ax.text(50, -0.4, 'MEDIUM', ha='center', va='top', fontsize=12, fontweight='bold', color='green')
        ax.text(100, -0.4, 'HIGH', ha='center', va='top', fontsize=12, fontweight='bold', color='red')
        ax.text(50, -0.9, 'Confidence on Predictions', ha='center', va='top', fontsize=13, fontweight='bold', color='black')

        if show_boxes and raw_heatmap is not None:
            ax.axvline(x=60, ymin=0, ymax=1, color='red', linestyle='--', linewidth=2)
            ax.text(60, 1.05, '← 60% (Bounding box threshold)', ha='left', va='bottom',
                   fontsize=9, color='red', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ============================================================
        # TOGGLE BUTTON FOR BOUNDING BOXES
        # ============================================================

        col_toggle1, col_toggle2, col_toggle3 = st.columns([2, 2, 2])
        with col_toggle1:
            pass
        with col_toggle2:
            if show_boxes:
                button_text = "🔴 Hide Bounding Boxes"
                button_help = "Click to remove the red boxes from the overlay image"
            else:
                button_text = "🔴 Show Bounding Boxes"
                button_help = "Click to display red boxes around high-confidence regions (>60%)"

            if st.button(button_text, key="toggle_boxes_batch", width="stretch", help=button_help):
                st.session_state.show_bounding_boxes = not show_boxes
                st.rerun()
        with col_toggle3:
            pass

        if show_boxes:
            st.success("✅ **Bounding boxes are currently VISIBLE.** The red boxes show high-confidence regions (>60%).")
        else:
            st.info("🔲 **Bounding boxes are currently HIDDEN.** Click 'Show Bounding Boxes' to see the high-confidence regions.")

        # ============================================================
        # STATISTICS AND EXPLANATION - WITH ADAPTIVE TEXT
        # ============================================================

        if show_boxes and raw_heatmap is not None and boxes and len(boxes) > 0:
            # Case 1: Bounding boxes found

            # Sort boxes by score (largest to smallest)
            sorted_boxes = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)

            # Build percentage list with box numbers
            percentage_items = []
            for idx, (box, score) in enumerate(sorted_boxes, 1):
                percentage_items.append(f"  Box {idx}: {score*100:.0f}%")
            percentages_text = "<br>".join(percentage_items)

            # Build confidence breakdown text
            high_count = len([s for s in scores if s >= 0.8])
            med_count = len([s for s in scores if 0.7 <= s < 0.8])
            low_count = len([s for s in scores if 0.6 <= s < 0.7])

            breakdown_parts = []
            if high_count > 0:
                breakdown_parts.append(f"🔴 {high_count} region(s) with >80% confidence")
            if med_count > 0:
                breakdown_parts.append(f"🟠 {med_count} region(s) with 70-80% confidence")
            if low_count > 0:
                breakdown_parts.append(f"🟡 {low_count} region(s) with 60-70% confidence")
            breakdown_text = " | ".join(breakdown_parts)

            # Build dynamic example and explanation based on number of boxes
            num_boxes = len(sorted_boxes)

            if num_boxes == 1:
                # SINGLE BOX - Simplified explanation
                stats_content = [
                    f"📍 1 high-confidence region identified.",
                    "",
                    f"The RED box on the overlay image shows the exact image area that most strongly influenced the diagnosis of {predicted_class}.",
                    "",
                    f"📊 **Model Confidence:** {sorted_boxes[0][1]*100:.0f}%",
                    "",
                    f"📈 **Confidence category:** 🔴 >80% confidence.",
                    "",
                    "💡 **What this percentage means:**",
                    f"The AI is {sorted_boxes[0][1]*100:.0f}% confident that the area inside the RED box shows signs of {predicted_class}.",
                    "",
                    "✅ **Higher percentages = Stronger evidence** that the disease is present in that specific area."
                ]
            else:
                # MULTIPLE BOXES - Full explanation
                stats_content = [
                    f"📍 {len(boxes)} high-confidence regions identified.",
                    "",
                    f"These RED boxes on the overlay image show the exact image areas that most strongly influenced the diagnosis of {predicted_class}.",
                    "",
                    "📊 **Model Confidence breakdown (highest to lowest):**",
                    percentages_text,
                    "",
                    f"📈 **Model Confidence categories:** {breakdown_text}.",
                    "",
                    "💡 **What these percentages mean:**",
                    "Each percentage tells you how confident the AI is that the area inside or above THAT SPECIFIC BOX shows signs of the disease.",
                    "",
                    "📌 **Important:** These percentages are independent of each other. They do NOT add up to 100%. Each box has its own confidence score.",
                    "",
                    f"🔍 **For example:** The AI is {sorted_boxes[0][1]*100:.0f}% confident the area in Box 1 shows {predicted_class}. It is {sorted_boxes[-1][1]*100:.0f}% confident the area in Box {num_boxes} shows the same disease. Each box is assessed separately.",
                    "",
                    "✅ **Higher percentages = Stronger evidence** that the disease is present in that specific area."
                ]

            stats_html = "<br>".join(stats_content)

            html_expander(
                title="VIEW DETAILED ANALYSIS OF MODEL'S FOCUS AREAS (Click to Expand)",
                content_html=f'<div class="reference-text">{stats_html}</div>',
                icon="📊"
            )

        elif show_boxes and raw_heatmap is not None and confidence_value >= 0.7:
            # Case 2: No bounding boxes but high confidence - diffuse disease explanation
            stats_content = [
                f"📍 No RED boxes were drawn for this image.",
                "",
                "📦 **Why no boxes?** A RED box is only drawn when there is a contiguous region that is:",
                "   • At least 1% of the total image size, AND",
                "   • Has a confidence score >60%",
                "",
                f"📊 **Global Confidence:** {confidence_value*100:.1f}%",
                "",
                "🔍 **What this means:**",
                f"{predicted_class} often has diffuse symptoms - the disease evidence is spread across the image rather than concentrated in one large spot. The model recognises the overall pattern of damage across the whole image, even though no single region meets the size threshold for a box.",
                "",
                "💡 Think of it like this: You can recognise a forest from a distance (high global confidence) without being able to point to a single tree that defines it (local bounding boxes).",
                "",
                "📌 **Note:** The coloured overlay still shows the confidence pattern. Warmer colours (red/orange/yellow) indicate areas where the AI had more confidence about the diagnosis, even if they don't meet the size threshold for a box."
            ]

            stats_html = "<br>".join(stats_content)

            html_expander(
                title="VIEW DETAILED ANALYSIS OF MODEL'S FOCUS AREAS (Click to Expand)",
                content_html=f'<div class="reference-text">{stats_html}</div>',
                icon="📊"
            )

        elif not show_boxes:
            stats_content = [
                "🔲 Bounding boxes are currently HIDDEN.",
                "",
                "Click 'Show Bounding Boxes' above to see the red boxes on the overlay image."
            ]

            stats_html = "<br>".join(stats_content)

            html_expander(
                title="VIEW DETAILED ANALYSIS OF MODEL'S FOCUS AREAS (Click to Expand)",
                content_html=f'<div class="reference-text">{stats_html}</div>',
                icon="📊"
            )

        else:
            stats_content = [
                "📍 No high-confidence regions above 60% threshold.",
                "",
                "The disease evidence is distributed across the image rather than concentrated in specific spots. The heatmap colours show the overall confidence pattern."
            ]

            stats_html = "<br>".join(stats_content)

            html_expander(
                title="VIEW DETAILED ANALYSIS OF MODEL'S FOCUS AREAS (Click to Expand)",
                content_html=f'<div class="reference-text">{stats_html}</div>',
                icon="📊"
            )

    # ============================================================
    # GRAD-CAM COLOUR LEGEND
    # ============================================================
    st.markdown("""
<div class="section-card">
<h3>📊 LEGEND FOR THE VISUAL OVERLAY COLOURS</h3>
<p>   🔴 <strong>RED (HOT)</strong>     = HIGH confidence (>80%) - These areas strongly indicate the disease</p>
<p>   🟠 <strong>ORANGE</strong>        = HIGH-MEDIUM confidence (70-80%)</p>
<p>   🟡 <strong>YELLOW</strong>        = MEDIUM confidence (60-70%)</p>
<p>   🟢 <strong>GREEN</strong>         = LOW-MEDIUM confidence (40-60%)</p>
<p>   💠 <strong>CYAN</strong>          = VERY LOW confidence (20-40%)</p>
<p>   🔵 <strong>BLUE (COOL)</strong>   = LOWEST confidence (<20%) - These areas had minimal impact on the diagnosis</p>
<p class="gradcam-tip">💡 The warmer the colour (Red → Orange → Yellow), the more confident the AI is that the disease is present. Cooler colours (Green → Cyan → Blue) show areas the AI is less confident about.</p>
<p class="gradcam-tip">📦 <strong>RED BOUNDING BOXES:</strong> Red boxes are drawn directly on the overlay image to highlight regions with >60% confidence.
   The percentage on each red box shows the AI's confidence level for that specific area. These are the specific areas the model relied on the most to make its diagnosis.</p>
</div>
""", unsafe_allow_html=True)

    # XAI Sources
    xai_refs = []
    for ref_num in treatment.get('xai_ref_numbers', []):
        if ref_num in get_references():
            xai_refs.append(f"[{len(xai_refs)+1}] {get_references()[ref_num]}")

    if xai_refs:
        references_html = "<br>".join(xai_refs)
        html_expander(
            title="VIEW XAI SOURCES (Click to Expand)",
            content_html=f'<div class="reference-text">{references_html}</div>',
            icon="📚"
        )

    # ============================================================
    # TREATMENT RECOMMENDATION (if not healthy)
    # ============================================================
    if not treatment.get('is_healthy', False):
        # Determine urgency based on confidence
        if confidence_value >= 0.9:
            urgency_icon = "🔴"
            urgency_level = "URGENT"
            action = "IMMEDIATE TREATMENT REQUIRED"
            confidence_desc = "VERY HIGH"
            confidence_icon = "🟢"
            reason = "Very clear disease symptoms - act immediately"
        elif confidence_value >= 0.8:
            urgency_icon = "🟠"
            urgency_level = "HIGH"
            action = "TREAT PROMPTLY"
            confidence_desc = "HIGH"
            confidence_icon = "🟢"
            reason = "Clear disease symptoms visible - treat promptly"
        elif confidence_value >= 0.7:
            urgency_icon = "🟡"
            urgency_level = "MEDIUM"
            action = "TREAT PROMPTLY"
            confidence_desc = "HIGH"
            confidence_icon = "🟡"
            reason = "Clear but not extremely strong indicators"
        elif confidence_value >= 0.6:
            urgency_icon = "🟡"
            urgency_level = "MEDIUM"
            action = "CONSIDER TREATMENT"
            confidence_desc = "MODERATE"
            confidence_icon = "🟡"
            reason = "Some disease symptoms visible - consider treatment"
        elif confidence_value >= 0.5:
            urgency_icon = "🟠"
            urgency_level = "CAUTIOUS"
            action = "VERIFY BEFORE TREATMENT"
            confidence_desc = "MODERATE"
            confidence_icon = "🟠"
            reason = "Weak disease indicators present"
        elif confidence_value >= 0.4:
            urgency_icon = "🟠"
            urgency_level = "CAUTIOUS"
            action = "VERIFY BEFORE TREATMENT"
            confidence_desc = "LOW"
            confidence_icon = "🟠"
            reason = "Unclear or mixed indicators"
        elif confidence_value >= 0.3:
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

        st.markdown(f"""
<div class="section-card">
<h3>🌾 TREATMENT RECOMMENDATION FOR: {predicted_class}</h3>
<p><strong>📊 DIAGNOSIS:</strong> {predicted_class}</p>
<p><strong>🏷️ TYPE:</strong> {treatment['category']}</p>
<p><strong>📈 CLASSIFIER CONFIDENCE:</strong> {confidence_value*100:.1f}% ({confidence_desc}) {confidence_icon}</p>
<p><strong>⚡ URGENCY & ACTION:</strong> {urgency_icon} {urgency_level} - {action}</p>
<p><strong>💡 REASON:</strong> {reason}</p>
</div>
""", unsafe_allow_html=True)

        # Urgency Key
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

        # Causes & Characteristics
        st.markdown(f"""
<div class="section-card">
<h3>📋 CAUSES & CHARACTERISTICS</h3>
<p style="white-space: pre-line;">{treatment.get('causes_characteristics', 'Information not available')}</p>
</div>
""", unsafe_allow_html=True)

        # Recommended Management
        st.markdown(f"""
<div class="section-card">
<h3>🔧 RECOMMENDED MANAGEMENT</h3>
<p style="white-space: pre-line;">{treatment['management']}</p>
</div>
""", unsafe_allow_html=True)

        # Chemical Control
        st.markdown(f"""
<div class="section-card">
<h3>💊 CHEMICAL CONTROL</h3>
<p style="white-space: pre-line;">{treatment['chemical_control']}</p>
</div>
""", unsafe_allow_html=True)

        # MANAGEMENT REFERENCES
        management_refs = treatment.get('management_refs', [])
        if management_refs:
            management_refs_text = format_reference_list_sequential(management_refs, get_references())
            management_refs_html = management_refs_text.replace("\n", "<br>")
            html_expander(
                title="VIEW MANAGEMENT REFERENCES (Click to Expand)",
                content_html=f'<div class="reference-text">{management_refs_html}</div>',
                icon="📚"
            )

        # CHEMICAL REFERENCES
        chemical_refs_original = treatment.get('chemical_refs_original', [])
        if chemical_refs_original:
            chem_ref_list = []
            for idx, ref_num in enumerate(chemical_refs_original, 1):
                if ref_num in get_references():
                    chem_ref_list.append(f"[{idx}] {get_references()[ref_num]}")
                else:
                    chem_ref_list.append(f"[{idx}] Reference {ref_num}")
            chemical_refs_text = "\n".join(chem_ref_list)
        else:
            chemical_refs_text = "See product labels for specific chemical references"

        chemical_refs_html = chemical_refs_text.replace("\n", "<br>")
        html_expander(
            title="VIEW CHEMICAL REFERENCES (Click to Expand)",
            content_html=f'<div class="reference-text">{chemical_refs_html}</div>',
            icon="📚"
        )

        # Important Notes
        st.markdown(f"""
<div class="section-card">
<h3>📋 IMPORTANT NOTES:</h3>
<p>  • <strong>Diagnosis:</strong> AI-assisted with {confidence_value*100:.1f}% confidence. Confirm with expert if uncertain.</p>
<p>  • <strong>Treatments:</strong> Curated from verified sources (product labels, research papers, extension guides)</p>
<p>  • <strong>Always read product labels</strong> before applying any chemicals</p>
<p>  • <strong>Local availability:</strong> Check with your local agrovet for product availability</p>
<p>  • <strong>Expert consultation:</strong> Contact your local agricultural extension officer for confirmation</p>
</div>
""", unsafe_allow_html=True)

        # Export and Share buttons
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("📄 Export Report for This Image", width="stretch"):
                single_report_data = {
                    'class': predicted_class,
                    'confidence': confidence_value
                }
                report = generate_export_report(single_report_data, treatment, get_references(), None)
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"crop_doctor_report_{selected_result['filename'].replace('.jpg', '')}_{datetime.now(eat_timezone).strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_selected_image"
                )
        with col_exp2:
            display_whatsapp_share_button(predicted_class, confidence_value, st.session_state.location)
    else:
        # Healthy crop display
        st.markdown(f"""
<div class="section-card">
<h3>🌿 HEALTHY CROP ASSESSMENT</h3>
<p><strong>Diagnosis:</strong> {predicted_class}</p>
<p><strong>Confidence:</strong> {confidence_value*100:.1f}%</p>
<p><strong>Category:</strong> {clean_category(treatment['category'])}</p>
<p>Your crop appears healthy. Continue good farming practices.</p>
</div>
""", unsafe_allow_html=True)

    # ============================================================
    # OPTIONS MENU
    # ============================================================
    #st.markdown("---")
    #st.markdown("## 💡 OPTIONS MENU")
    st.caption("Explore alternative diagnoses for this image. Click on any option below to replace the current view.")

    num_predictions = len(selected_result['top_predictions'])

    menu_html = f"""
<div class="section-card">
<h3>💡 OPTIONS MENU</h3>
<p><strong>1.</strong> Get treatment for PRIMARY DIAGNOSIS ({selected_result['top_predictions'][0]['class']}) - {float(selected_result['top_predictions'][0]['confidence'])*100:.1f}%</p>
"""
    for i in range(1, num_predictions):
        pred_conf = float(selected_result['top_predictions'][i]['confidence']) if hasattr(selected_result['top_predictions'][i]['confidence'], 'item') else selected_result['top_predictions'][i]['confidence']
        menu_html += f'<p><strong>{i+1}.</strong> Get treatment for ALTERNATIVE {i} ({selected_result["top_predictions"][i]["class"]}) - {pred_conf*100:.1f}%</p>\n'

    menu_html += f"""
<p><strong>{num_predictions + 1}.</strong> Show common chemicals for ALL TOP {num_predictions} diseases</p>
<p><strong>{num_predictions + 2}.</strong> Change number of top predictions (current: {st.session_state.current_top_k})</p>
<p><strong>{num_predictions + 3}.</strong> Exit batch view</p>
</div>
"""
    st.markdown(menu_html, unsafe_allow_html=True)

    st.info("💡 **Tip:** To analyse a different image from this batch, simply scroll up and select a different image from the dropdown menu at the top of this page.")

    total_options = num_predictions + 3
    cols = st.columns(min(total_options, 10))

    for i in range(1, min(total_options + 1, 11)):
        with cols[i-1]:
            if st.button(f"{i}", key=f"batch_menu_btn_{i}", width="stretch"):
                if i == total_options:
                    st.session_state.show_batch_results = False
                    st.session_state.batch_results = None
                    st.session_state.batch_mode = False
                    st.session_state.show_alternative_batch = None
                    st.session_state.batch_previous_selected_index = None
                    st.rerun()
                elif i == total_options - 1:
                    st.session_state.show_k_dialog = True
                    st.rerun()
                elif i == total_options - 2:
                    st.session_state.show_common_chemicals_batch = True
                    st.session_state.show_alternative_batch = None
                    st.rerun()
                elif 1 <= i <= num_predictions:
                    if i == 1:
                        st.session_state.show_alternative_batch = None
                    else:
                        st.session_state.show_alternative_batch = i - 1
                    st.session_state.show_common_chemicals_batch = False
                    st.rerun()

    if st.session_state.get('show_common_chemicals_batch', False):
        st.markdown("---")
        st.markdown("#### 🌿 Common Chemicals for All Top Predictions")
        all_top_preds = []
        for pred in selected_result['top_predictions']:
            pred_conf = float(pred['confidence']) if hasattr(pred['confidence'], 'item') else pred['confidence']
            all_top_preds.append({'class': pred['class'], 'confidence': pred_conf})
        show_common_chemicals_for_top_k(all_top_preds, get_references())
        if st.button("✖ Close", width="stretch", key="close_common_chemicals_batch"):
            st.session_state.show_common_chemicals_batch = False
            st.rerun()

    # ============================================================
    # VIEW ALL BATCH RESULTS (IMAGES AND THEIR OVERLAYS IN GALLERY)
    # ============================================================
    st.markdown("---")
    st.markdown("""
<div class="section-card">
<h3>📸 VIEW ALL BATCH RESULTS (IMAGES AND THEIR OVERLAYS IN GALLERY)</h3>
</div>
""", unsafe_allow_html=True)

    show_boxes_batch_view = st.session_state.get('show_bounding_boxes_batch_view', True)

    col_toggle_view1, col_toggle_view2, col_toggle_view3 = st.columns([2, 2, 2])
    with col_toggle_view1:
        pass
    with col_toggle_view2:
        if show_boxes_batch_view:
            button_text_view = "🔴 Hide Red Boxes in Gallery"
            button_help_view = "Click to remove red boxes from all gallery images"
        else:
            button_text_view = "🔴 Show Red Boxes in Gallery"
            button_help_view = "Click to display red boxes around high-confidence regions (>60%) for all gallery images"

        if st.button(button_text_view, key="toggle_boxes_batch_gallery", use_container_width=True, help=button_help_view):
            st.session_state.show_bounding_boxes_batch_view = not show_boxes_batch_view
            st.rerun()
    with col_toggle_view3:
        pass

    if show_boxes_batch_view:
        st.success("✅ **Red boxes in the gallery are currently VISIBLE.**")
    else:
        st.info("🔲 **Red boxes in the gallery are currently HIDDEN.** Click 'Show Red Boxes in Gallery' to see them.")

    with st.expander("\U0001F4F8 Click to expand and view all images", expanded=False):
        for idx, r in enumerate(successful):
            conf_val = float(r['primary_confidence']) if hasattr(r['primary_confidence'], 'item') else r['primary_confidence']
            st.markdown(f"**📷 {idx+1}. {r['filename']}** - {r['primary_diagnosis']} ({conf_val*100:.1f}%)")

            col1_img, col2_img = st.columns(2)
            with col1_img:
                st.image(r['image'], caption="Original Image", use_container_width=True)

            with col2_img:
                overlay_to_show = r.get('heatmap_overlay')
                raw_heatmap_exists = r.get('raw_heatmap') is not None

                if show_boxes_batch_view and raw_heatmap_exists and overlay_to_show is not None:
                    try:
                        overlay_with_boxes, boxes, scores = extract_bounding_boxes_from_heatmap(
                            r['raw_heatmap'], overlay_to_show, threshold=0.6, min_box_area=500
                        )

                        if boxes and len(boxes) == 1:
                            st.image(overlay_with_boxes,
                                    caption=f"Visual Overlay Image with a RED Box. ({len(boxes)} region where AI had >60% confidence about the diagnosis.)",
                                    use_container_width=True)
                        elif boxes and len(boxes) > 1:
                            st.image(overlay_with_boxes,
                                    caption=f"Visual Overlay Image with RED Boxes. ({len(boxes)} regions where AI had >60% confidence about the diagnosis.)",
                                    use_container_width=True)
                        else:
                            if conf_val >= 0.7:
                                st.image(overlay_to_show,
                                        caption=f"Visual Overlay Image (No single region where AI had >60% confidence about the diagnosis. Evidence is spread across the whole image.)",
                                        use_container_width=True)
                                st.caption("💡 The model is confident but disease signs are distributed across the image.")
                            else:
                                st.image(overlay_to_show, caption=f"Visual overlay image showing graduated influence of areas that led the model to pick on: \n{r['primary_diagnosis']}.", use_container_width=True)
                    except Exception as e:
                        st.image(overlay_to_show, caption=f"Visual overlay image showing graduated influence of areas that led the model to pick on: \n{r['primary_diagnosis']}.", use_container_width=True)
                        st.caption(f"⚠️ Bounding boxes temporarily unavailable.")
                else:
                    if overlay_to_show is not None:
                        st.image(overlay_to_show, caption=f"Visual overlay image showing graduated influence of areas that led the model to pick on: \n{r['primary_diagnosis']}.", use_container_width=True)
                    else:
                        st.caption("ℹ️ No overlay available for this image.")

            #st.markdown("---")

    # ============================================================
    # ONLINE MODE FEATURES
    # ============================================================
    if st.session_state.mode == "online":
        display_online_features(
            predicted_class,
            clean_category(treatment['category']),
            st.session_state.location,
            treatment
        )

    # ============================================================
    # FEEDBACK SECTION
    # ============================================================
    st.info("💡 **We value your feedback!** After exploring all the features above, please share your experience with us. Your answers help improve Crop Doctor for all Kenyan farmers.")
    display_feedback_section(predicted_class, confidence_value)

    # ============================================================
    # THANK YOU MESSAGE
    # ============================================================
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: #e8f5e9; border-radius: 15px;">
        <p style="font-size: 16px; margin-bottom: 5px;">🙏 <strong>Asante Sana Kwa Maoni Yako! (Thank You Very Much For Your Feedback!)</strong></p>
        <p style="font-size: 12px; color: #888; margin-top: 10px;">🌾 Happy Farming! 🌾</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("\u2139\uFE0F All timestamps are in East Africa Time (EAT, UTC+3)")


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
            if st.button("❓ Help / How to use the system", width="stretch" ):
                st.session_state.show_help = not st.session_state.show_help

            # Classes button
            if st.button("📋 List of Supported Classes", width="stretch" ):
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

            # ============================================================
            # PROCESSING MODE SELECTION (SINGLE vs BATCH)
            # ============================================================
            processing_mode = st.radio(
                "Select Processing Mode:",
                ["Single Image", "Batch Processing (Multiple Images)"],
                horizontal=True,
                key="processing_mode"
            )

            if processing_mode == "Single Image":
                st.caption("Upload one image at a time for detailed analysis")

                if st.button("📷 Take Photo", width="stretch" ):
                    st.session_state.camera_active = True

                if st.session_state.camera_active:
                    camera_image = st.camera_input("Take a photo", key="camera")
                    if camera_image:
                        st.session_state.current_image = Image.open(camera_image)
                        st.session_state.camera_active = False
                        st.session_state.batch_mode = False
                        st.rerun()

                uploaded_file = st.file_uploader(
                    "Or choose from gallery",
                    type=['jpg', 'jpeg', 'png'],
                    key="uploader"
                )
                if uploaded_file:
                    st.session_state.current_image = Image.open(uploaded_file)
                    st.session_state.batch_mode = False
                    st.session_state.batch_results = None
                    st.session_state.show_batch_results = False

                if st.session_state.current_image is not None:
                    st.image(st.session_state.current_image, caption="Selected Image", width="stretch")

                    if st.button("🔬 DIAGNOSE & RECOMMEND", type="primary", width="stretch"):
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
                            st.session_state.batch_mode = False
                            st.session_state.show_batch_results = False

                            # Clear previous alt data
                            st.session_state.current_alt_data = {}

                            # Save image for training (primary diagnosis only)
                            save_user_image_for_training(image, top_predictions[0]['class'], top_predictions[0]['confidence'], top_predictions)

                            # Generate Grad-CAM for ALL predictions
                            for alt_idx, pred in enumerate(top_predictions):
                                # Generate raw heatmap (values 0-1)
                                raw_heatmap = gradcam.generate_heatmap(img_array, pred['idx'])

                                # Generate coloured overlay for display
                                overlay = gradcam.overlay_heatmap(raw_heatmap, st.session_state.current_original_img)
                                treatment = get_full_treatment(pred['class'], references)

                                # Determine crop type
                                if 'maize' in pred['class'].lower():
                                    crop_type = "Maize"
                                elif 'beans' in pred['class'].lower():
                                    crop_type = "Beans"
                                else:
                                    crop_type = "Tomato"

                                # Save heatmap overlay
                                save_filename = f"gradcam_{pred['class'].replace(' ', '_').replace('/', '_')}_alt{alt_idx}.png"
                                save_path = os.path.join(os.getcwd(), save_filename)
                                cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                                # Store in session state - CRITICAL for alternatives
                                st.session_state.current_alt_data[alt_idx] = {
                                    'class': pred['class'],
                                    'confidence': pred['confidence'],
                                    'idx': pred['idx'],
                                    'treatment': treatment,
                                    'heatmap_overlay': overlay,
                                    'raw_heatmap': raw_heatmap,  # Store raw heatmap for bounding boxes
                                    'references': references,
                                    'original_img': st.session_state.current_original_img,
                                    'crop_type': crop_type,
                                    'save_path': save_path,
                                    'save_filename': save_filename
                                }

                                # Debug print
                                print(f"Stored alt_idx {alt_idx}: {pred['class']}")

                            st.rerun()

            else:  # Batch Processing Mode
                st.caption("📁 Upload multiple images for batch analysis (ideal for research)")
                st.info("💡 **Tip:** Batch processing is great for researchers, extension officers, and large farms. Upload multiple images at once. The more the images, the longer the analysis will take. A batch of 30 images takes about 2 minutes to process.")

                batch_files = st.file_uploader(
                    "Select multiple images",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                    key="batch_uploader"
                )

                if batch_files:
                    st.markdown(f"**Selected {len(batch_files)} images**")

                    # Preview thumbnails
                    st.markdown("#### 📸 Image Preview")
                    preview_cols = st.columns(min(5, len(batch_files)))
                    for idx, file in enumerate(batch_files[:5]):
                        with preview_cols[idx]:
                            img = Image.open(file)
                            img.thumbnail((100, 100))
                            st.image(img, caption=file.name[:15], width="stretch")

                    if len(batch_files) > 5:
                        st.caption(f"... and {len(batch_files) - 5} more images")

                    # Warning for large batches
                    if len(batch_files) > 30:
                        st.warning("⚠️ Large batch detected. Processing may take several minutes. Please be patient.")

                    if st.button("🔬 PROCESS BATCH", type="primary", width="stretch" ):
                        with st.spinner(f"Processing {len(batch_files)} images... This may take a few minutes."):
                            results = process_batch_images(batch_files, model, class_names, references, gradcam)
                            st.session_state.batch_results = results
                            st.session_state.show_batch_results = True
                            st.session_state.batch_mode = True
                            st.session_state.current_image = None
                            st.session_state.show_results = False
                            st.rerun()
                else:
                    st.info("👈 Select one or more images to begin batch processing")

        with right_col:
            # ============================================================
            # BATCH RESULTS DISPLAY
            # ============================================================
            if st.session_state.get('show_batch_results', False) and st.session_state.get('batch_results'):
                display_batch_results(st.session_state.batch_results)

                col1_clear, col2_clear = st.columns(2)
                with col1_clear:
                    if st.button("✖ Clear Batch Results", width="stretch" ):
                        st.session_state.show_batch_results = False
                        st.session_state.batch_results = None
                        st.session_state.batch_mode = False
                        st.rerun()
                with col2_clear:
                    if st.button("🔄 New Batch", width="stretch" ):
                        st.session_state.show_batch_results = False
                        st.session_state.batch_results = None
                        st.session_state.batch_mode = False
                        st.rerun()

            # ============================================================
            # SINGLE IMAGE RESULTS DISPLAY
            # ============================================================
            elif st.session_state.show_results and st.session_state.current_top_predictions:
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
                        current_confidence = disease_data['confidence']

                        display_top_predictions(top_predictions)
                        display_xai_analysis(disease_data)

                        # Only show treatment recommendation for diseased crops
                        if not disease_data['treatment'].get('is_healthy', False):
                            display_treatment_recommendation(disease_data['treatment'], references, disease_data['confidence'])

                        # Export Report Button and WhatsApp Share
                        col1_export, col2_export, col3_whatsapp = st.columns(3)
                        with col1_export:
                            if st.button("📄 Export Report", width="stretch" , key="export_alt"):
                                report = generate_export_report(disease_data, disease_data['treatment'], references, None)
                                from datetime import datetime, timedelta, timezone as dt_timezone
                                eat_timezone = dt_timezone(timedelta(hours=3))
                                local_now = datetime.now(eat_timezone)
                                local_timestamp = local_now.strftime('%Y%m%d_%H%M%S')

                                st.download_button(
                                    label="📥 Download Report",
                                    data=report,
                                    file_name=f"crop_doctor_report_{local_timestamp}.txt",
                                    mime="text/plain",
                                    key="download_alt"
                                )

                        with col3_whatsapp:
                            display_whatsapp_share_button(current_disease, current_confidence, st.session_state.location)

                        # OPTIONS MENU
                        display_options_menu(top_predictions, references, st.session_state.location, class_names, current_disease, current_crop_type, current_treatment)

                        # INVITATION MESSAGE
                        #st.markdown("---")
                        st.info("💡 **We value your feedback!** After exploring all the features above, please share your experience with us. Your answers help improve Crop Doctor for all Kenyan farmers.")

                        # FEEDBACK SECTION
                        display_feedback_section(current_disease, current_confidence)

                        # THANK YOU MESSAGE
                        #st.markdown("---")
                        st.markdown("""
                        <div style="text-align: center; padding: 20px; background: #e8f5e9; border-radius: 15px;">
                            <p style="font-size: 16px; margin-bottom: 5px;">🙏 <strong>Asante Sana Kwa Maoni Yako! (Thank You Very Much For Your Feedback!)</strong></p>
                            <p style="font-size: 12px; color: #888; margin-top: 10px;">🌾 Happy Farming! 🌾</p>
                        </div>
                        """, unsafe_allow_html=True)

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
                    current_confidence = primary_data['confidence']

                    display_top_predictions(top_predictions)
                    display_xai_analysis(primary_data)

                    # Only show treatment recommendation for diseased crops
                    if not primary_data['treatment'].get('is_healthy', False):
                        display_treatment_recommendation(primary_data['treatment'], references, primary_data['confidence'])

                    # Export Report Button and WhatsApp Share
                    col1_export, col2_export, col3_whatsapp = st.columns(3)
                    with col1_export:
                        if st.button("📄 Export Report", width="stretch" , key="export_primary"):
                            report = generate_export_report(primary_data, primary_data['treatment'], references, None)
                            from datetime import datetime, timedelta, timezone as dt_timezone
                            eat_timezone = dt_timezone(timedelta(hours=3))
                            local_now = datetime.now(eat_timezone)
                            local_timestamp = local_now.strftime('%Y%m%d_%H%M%S')

                            st.download_button(
                                label="📥 Download Report",
                                data=report,
                                file_name=f"crop_doctor_report_{local_timestamp}.txt",
                                mime="text/plain",
                                key="download_primary"
                            )

                    with col3_whatsapp:
                        display_whatsapp_share_button(current_disease, current_confidence, st.session_state.location)

                    # OPTIONS MENU
                    display_options_menu(top_predictions, references, st.session_state.location, class_names, current_disease, current_crop_type, current_treatment)

                    # INVITATION MESSAGE
                    st.markdown("---")
                    st.info("💡 **We value your feedback!** After exploring all the features above, please share your experience with us. Your answers help improve Crop Doctor for all Kenyan farmers.")

                    # FEEDBACK SECTION
                    display_feedback_section(current_disease, current_confidence)

                    # THANK YOU MESSAGE
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; padding: 20px; background: #e8f5e9; border-radius: 15px;">
                        <p style="font-size: 16px; margin-bottom: 5px;">🙏 <strong>Asante Sana Kwa Maoni Yako! (Thank You Very Much For Your Feedback!)</strong></p>
                        <p style="font-size: 12px; color: #888; margin-top: 10px;">🌾 Happy Farming! 🌾</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                if not st.session_state.get('batch_mode', False):
                    st.info("👈 Upload an image and click 'DIAGNOSE & RECOMMEND', or use Batch Processing for multiple images")

if __name__ == "__main__":
    main()
