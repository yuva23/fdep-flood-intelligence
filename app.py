import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
from openai import OpenAI
import json
import os
import datetime

# --- 1. CONFIGURATION & AUTH ---
st.set_page_config(page_title="FDEP Flood Intelligence", layout="wide")

def auth_ee():
    try:
        # Try initializing with default credentials
        ee.Initialize()
    except Exception:
        # If that fails, manually rebuild credentials from Streamlit Secrets
        if "EARTHENGINE_TOKEN" in st.secrets:
            credentials_path = os.path.expanduser("~/.config/earthengine/")
            os.makedirs(credentials_path, exist_ok=True)
            token_content = st.secrets["EARTHENGINE_TOKEN"]
            if not isinstance(token_content, str):
                token_content = json.dumps(token_content)
            with open(os.path.join(credentials_path, "credentials"), "w") as f:
                f.write(token_content)
            ee.Initialize()
        else:
            raise

auth_ee()

# --- 2. DATA ARCHIVE ---
flood_archive = {
    "2024": {
        "San Diego Flash Floods (Jan)": {"lat": 32.71, "lon": -117.16, "dates": ["2023-12-01", "2024-01-15", "2024-01-22", "2024-01-30"]},
        "Houston Floods (May)": {"lat": 29.76, "lon": -95.36, "dates": ["2024-04-01", "2024-04-15", "2024-05-02", "2024-05-10"]}
    },
    "2023": {
        "Libya Dam Collapse (Sep)": {"lat": 32.76, "lon": 22.63, "dates": ["2023-08-01", "2023-09-01", "2023-09-12", "2023-09-20"]}
    },
    "2022": {
        "Hurricane Ian (Florida)": {"lat": 26.64, "lon": -81.87, "dates": ["2022-09-01", "2022-09-15", "2022-09-29", "2022-10-05"]},
        "Pakistan Floods (Sindh)": {"lat": 26.90, "lon": 68.10, "dates": ["2022-08-01", "2022-08-10", "2022-08-20", "2022-08-30"]},
        "California Atmospheric River": {"lat": 38.58, "lon": -121.49, "dates": ["2022-12-01", "2022-12-15", "2023-01-05", "2023-01-15"]}
    }
}

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("FDEP Flood Intelligence")

st.sidebar.header("1. Select Event")
selected_year = st.sidebar.selectbox("Year", list(flood_archive.keys()), index=2)
event_list = list(flood_archive[selected_year].keys())
selected_event_name = st.sidebar.selectbox("Event", event_list)

# Load Params
params = flood_archive[selected_year][selected_event_name]
lat, lon = params["lat"], params["lon"]

# Helper to force dates to simple strings
def make_date_obj(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

default_dates = [make_date_obj(x) for x in params["dates"]]

with st.sidebar.expander("Date Settings", expanded=False):
    col1, col2 = st.columns(2)
    d1 = col1.date_input("Before Start", default_dates[0])
    d2 = col2.date_input("Before End", default_dates[1])
    d3 = col1.date_input("After Start", default_dates[2])
    d4 = col2.date_input("After End", default_dates[3])

st.sidebar.header("2. Sensor & Layers")
sensor_type = st.sidebar.radio("Satellite", ["Sentinel-1 (Radar)", "Sentinel-2 (Optical)"])
show_fdep = st.sidebar.checkbox("Overlay FDEP Conservation Lands", value=False)

# --- 4. EXECUTION ---
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

if st.sidebar.button("Run Analysis", type="primary"):
    st.session_state.analysis_active = True

if st.session_state.analysis_active:
    st.subheader(f"Analysis: {selected_event_name}")
    
    with st.spinner('Processing Satellite Data...'):
        roi = ee.Geometry.Point([lon, lat]).buffer(10000)
        m = geemap.Map(center=[lat, lon], zoom=10)
        flooded_ha = 0
        
        # Date Strings for GEE
        start_b_str = d1.strftime("%Y-%m-%d")
        end_b_str = d2.strftime("%Y-%m-%d")
        start_a_str = d3.strftime("%Y-%m-%d")
        end_a_str = d4.strftime("%Y-%m-%d")

        # FDEP Layer
        if show_fdep:
            try:
                fdep_url = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/DSL_Cons_Lands/MapServer"
                m.add_esri_layer(fdep_url, name="FDEP Conservation Lands", opacity=0.6)
            except:
                pass

        # SENTINEL-1 (RADAR)
        if sensor_type == "Sentinel-1 (Radar)":
            def get_sar(start, end):
                return (ee.ImageCollection('COPERNICUS/S1_GRD')
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))
                        .filterBounds(roi)
                        .filterDate(start, end)
                        .mosaic().clip(roi))

            before = get_sar(start_b_str, end_b_str)
            after = get_sar(start_a_str, end_a_str)

            # Detect Flood
            diff = after.focal_mean(50).divide(before.focal_mean(50))
            flood_mask = diff.select('VV').lt(0.8)
            flood_final = flood_mask.updateMask(before.select('VV').gt(-15)).selfMask()
            
            m.add_layer(before, {'min': -25, 'max': 0}, 'Before Storm')
            m.add_layer(after, {'min': -25, 'max': 0}, 'After Storm')
            m.add_layer(flood_final, {'palette': ['red']}, 'FLOOD DETECTED')
            
            # Stats
            area = flood_final.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e9
            ).getInfo()
            flooded_ha = area.get('VV', 0) / 10000
            
            # Download Button
            try:
                url = flood_final.getDownloadURL({'name': 'flood_map', 'scale': 30, 'region': roi})
                st.sidebar.markdown(f"[Download GeoTIFF]({url})")
            except: pass

        # SENTINEL-2 (OPTICAL)
        else:
            def get_opt(start, end):
                return (ee.ImageCollection('COPERNICUS/S2_SR')
                        .filterBounds(roi)
                        .filterDate(start, end)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        .median().clip(roi))
            
            before = get_opt(start_b_str, end_b_str)
            after = get_opt(start_a_str, end_a_str)
            vis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
            m.add_layer(before, vis, 'Before (Optical)')
            m.add_layer(after, vis, 'After (Optical)')

    m.to_streamlit(height=600)
    
    if sensor_type == "Sentinel-1 (Radar)":
        st.success(f"Detected Flood Extent: {flooded_ha:.2f} Hectares")

    # --- AI SECTION ---
    st.divider()
    st.subheader("AI Situation Report")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about this event..."):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        context = f"Event: {selected_event_name}. Flooded: {flooded_ha:.2f} ha. User Question: {prompt}"
        
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": context}, {"role": "user", "content": prompt}])
            reply = response.choices[0].message.content
            st.chat_message("assistant").write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"AI Error: {e}")
