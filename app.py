import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
from openai import OpenAI
from geopy.geocoders import Nominatim # The GPS Tool
import json
import os

# --- 1. CONFIGURATION & AUTH ---
st.set_page_config(page_title="Global Flood Intelligence", layout="wide")

def auth_ee():
    try:
        ee.Initialize(project='flood-intelligence-gee-12345')
    except Exception:
        if "EARTHENGINE_TOKEN" in st.secrets:
            credentials_path = os.path.expanduser("~/.config/earthengine/")
            os.makedirs(credentials_path, exist_ok=True)
            token_content = st.secrets["EARTHENGINE_TOKEN"]
            if not isinstance(token_content, str):
                token_content = json.dumps(token_content)
            with open(os.path.join(credentials_path, "credentials"), "w") as f:
                f.write(token_content)
            ee.Initialize(project='flood-intelligence-gee-12345')
        else:
            raise

auth_ee()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("🌍 Global Flood Intelligence")
st.sidebar.info("Sentinel-1 SAR | Multi-Region Support")

# GLOBAL SEARCH BAR
location_query = st.sidebar.text_input("Enter ANY Location (City, Country)", "Fort Myers, FL")

# Geocoding Logic (The "GPS" Search)
geolocator = Nominatim(user_agent="flood_app")
location = geolocator.geocode(location_query)

if location:
    lat = location.latitude
    lon = location.longitude
    st.sidebar.success(f"📍 Found: {location.address}")
    st.sidebar.caption(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
else:
    st.sidebar.error("Location not found! Defaulting to Fort Myers.")
    lat = 26.64
    lon = -81.87

# Date Selection
st.sidebar.subheader("Analysis Window")
before_start = st.sidebar.date_input("Before Start", value=pd.to_datetime("2022-09-01"))
before_end = st.sidebar.date_input("Before End", value=pd.to_datetime("2022-09-15"))
after_start = st.sidebar.date_input("After Start", value=pd.to_datetime("2022-09-29"))
after_end = st.sidebar.date_input("After End", value=pd.to_datetime("2022-10-05"))

# --- 3. SESSION STATE ---
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

if st.sidebar.button("Run Global Analysis"):
    st.session_state.analysis_active = True

# --- 4. CORE LOGIC ---
if st.session_state.analysis_active:
    with st.spinner(f'Scanning Satellite Data for {location_query}...'):
        roi = ee.Geometry.Point([lon, lat]).buffer(20000)
        
        def get_sar_layer(start, end, roi):
            return (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filterBounds(roi)
                    .filterDate(str(start), str(end))
                    .mosaic().clip(roi))

        before = get_sar_layer(before_start, before_end, roi)
        after = get_sar_layer(after_start, after_end, roi)

        smooth_radius = 50
        before_filtered = before.focal_mean(smooth_radius, 'circle', 'meters')
        after_filtered = after.focal_mean(smooth_radius, 'circle', 'meters')
        difference = after_filtered.divide(before_filtered)
        
        THRESHOLD = 0.8
        flood_mask = difference.select('VV').lt(THRESHOLD)
        perm_water = before_filtered.select('VV').gt(-15)
        flood_final = flood_mask.updateMask(perm_water).selfMask() 

        # Statistics
        flood_area = flood_final.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        flooded_ha = flood_area.get('VV', 0) / 10000

    # --- 5. VISUALIZATION ---
    st.subheader(f"Flood Analysis: {location_query}")
    
    # Map now centers on the SEARCHED location
    m = geemap.Map(center=[lat, lon], zoom=11)
    m.add_layer(before_filtered, {'min': -25, 'max': 0}, 'Before (SAR)')
    m.add_layer(after_filtered, {'min': -25, 'max': 0}, 'After (SAR)')
    m.add_layer(flood_final, {'palette': ['red']}, 'FLOOD EXTENT')
    m.to_streamlit(height=600)
    
    st.success(f"🛑 Detected Flood Extent: {flooded_ha:.2f} Hectares")
    
    # --- 6. AI ASSISTANT ---
    st.divider()
    st.subheader(" Global AI Impact Assessment")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about this area..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Enhanced Prompt for Global Context
        system_context = f"""
        You are an expert International Flood Response Analyst.
        Target Area: {location_query} (Lat: {lat}, Lon: {lon}).
        Data: {flooded_ha:.2f} hectares flooded.
        User Question: {prompt}
        
        If the location is a major city, mention specific infrastructure that might be at risk based on the coordinates.
        """

        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_context},
                    {"role": "user", "content": prompt}
                ]
            )
            ai_response = response.choices[0].message.content
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        except Exception as e:
            st.error(f"AI Error: {e}")

else:
    st.write("👈 Enter a city name (e.g., 'London' or 'Mumbai') and click 'Run Global Analysis'.")
