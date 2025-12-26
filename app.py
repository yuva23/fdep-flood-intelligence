import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
from openai import OpenAI
from geopy.geocoders import Nominatim
import json
import os
import datetime

# --- 1. CONFIGURATION & AUTH ---
st.set_page_config(page_title="FDEP Flood Intelligence Platform", layout="wide")

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

# --- 2. THE FLOOD ARCHIVE (DATA DICTIONARY) ---
# This serves as your "Database" of events
flood_archive = {
    "2024": {
        "San Diego Flash Floods (Jan)": {"lat": 32.71, "lon": -117.16, "dates": ["2023-12-01", "2024-01-15", "2024-01-22", "2024-01-30"]},
        "Houston Floods (May)": {"lat": 29.76, "lon": -95.36, "dates": ["2024-04-01", "2024-04-15", "2024-05-02", "2024-05-10"]}
    },
    "2023": {
        "Fort Lauderdale Flash Flood (Apr)": {"lat": 26.12, "lon": -80.14, "dates": ["2023-03-01", "2023-04-01", "2023-04-13", "2023-04-20"]},
        "Vermont Floods (Jul)": {"lat": 44.26, "lon": -72.57, "dates": ["2023-06-01", "2023-07-01", "2023-07-11", "2023-07-20"]},
        "Libya Dam Collapse (Sep)": {"lat": 32.76, "lon": 22.63, "dates": ["2023-08-01", "2023-09-01", "2023-09-12", "2023-09-20"]}
    },
    "2022": {
        "Hurricane Ian (Florida)": {"lat": 26.64, "lon": -81.87, "dates": ["2022-09-01", "2022-09-15", "2022-09-29", "2022-10-05"]},
        "Pakistan Floods (Sindh)": {"lat": 26.90, "lon": 68.10, "dates": ["2022-08-01", "2022-08-10", "2022-08-20", "2022-08-30"]},
        "Yellowstone Floods (Jun)": {"lat": 45.03, "lon": -110.70, "dates": ["2022-05-01", "2022-06-01", "2022-06-13", "2022-06-25"]}
    },
    "2021": {
        "Hurricane Ida (Louisiana)": {"lat": 29.59, "lon": -90.71, "dates": ["2021-08-01", "2021-08-20", "2021-08-30", "2021-09-10"]},
        "Germany/Belgium Floods": {"lat": 50.47, "lon": 6.85, "dates": ["2021-06-01", "2021-07-10", "2021-07-15", "2021-07-25"]}
    },
    "2020": {
        "Hurricane Sally (Pensacola)": {"lat": 30.42, "lon": -87.21, "dates": ["2020-08-15", "2020-09-10", "2020-09-17", "2020-09-25"]},
        "Midland Dam Failure (Michigan)": {"lat": 43.61, "lon": -84.22, "dates": ["2020-05-01", "2020-05-15", "2020-05-20", "2020-05-30"]}
    }
}

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Flag_of_Florida.svg/320px-Flag_of_Florida.svg.png", width=50)
st.sidebar.title("FDEP Flood Intelligence")

# A. HIERARCHICAL DROPDOWNS
st.sidebar.header("1. Event Selection")
selected_year = st.sidebar.selectbox("Select Year", list(flood_archive.keys()))
event_list = list(flood_archive[selected_year].keys())
selected_event_name = st.sidebar.selectbox("Select Major Event", event_list)

# Load Params
params = flood_archive[selected_year][selected_event_name]
lat, lon = params["lat"], params["lon"]
d = [pd.to_datetime(x).date() for x in params["dates"]]

# B. DATE & LOCATION OVERRIDE
with st.sidebar.expander("Advanced Settings / Custom Dates"):
    location_query = st.text_input("Location Name", selected_event_name)
    col1, col2 = st.columns(2)
    before_start = col1.date_input("Before Start", d[0])
    before_end = col2.date_input("Before End", d[1])
    after_start = col1.date_input("After Start", d[2])
    after_end = col2.date_input("After End", d[3])

# C. SENSOR & LAYERS
st.sidebar.header("2. Sensor & Layers")
sensor_type = st.sidebar.radio("Select Satellite", ["Sentinel-1 (Radar)", "Sentinel-2 (Optical)"])
show_fdep = st.sidebar.checkbox("Overlay FDEP Conservation Lands", value=False)

# --- 4. EXECUTION ---
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

if st.sidebar.button("Run Analysis", type="primary"):
    st.session_state.analysis_active = True

if st.session_state.analysis_active:
    # Main Dashboard Area
    st.subheader(f"Analysis: {selected_event_name} ({selected_year})")
    
    with st.spinner('Accessing Satellite Data & FDEP Servers...'):
        roi = ee.Geometry.Point([lon, lat]).buffer(20000)
        m = geemap.Map(center=[lat, lon], zoom=11)
        flooded_ha = 0
        
        def dstr(d): return d.strftime("%Y-%m-%d")

        # --- FDEP INTEGRATION (THE CHERRY ON TOP) ---
        if show_fdep:
            try:
                # FDEP Map Direct Public Service (Conservation Lands)
                fdep_url = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/DSL_Cons_Lands/MapServer"
                m.add_esri_layer(fdep_url, name="FDEP Conservation Lands", opacity=0.6)
                st.toast("FDEP Data Layer Loaded Successfully!", icon="✅")
            except:
                st.warning("Could not connect to FDEP Map Direct Server. Showing standard map.")

        # --- SATELLITE PROCESSING ---
        if sensor_type == "Sentinel-1 (Radar)":
            collection = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.eq('instrumentMode', 'IW')).filterBounds(roi)
            before = collection.filterDate(dstr(before_start), dstr(before_end)).mosaic().clip(roi)
            after = collection.filterDate(dstr(after_start), dstr(after_end)).mosaic().clip(roi)

            # Detect Flood
            diff = after.focal_mean(50).divide(before.focal_mean(50))
            flood_mask = diff.select('VV').lt(0.8)
            flood_final = flood_mask.updateMask(before.select('VV').gt(-15)).selfMask()
            
            m.add_layer(before, {'min': -25, 'max': 0}, 'Before Storm')
            m.add_layer(after, {'min': -25, 'max': 0}, 'After Storm')
            m.add_layer(flood_final, {'palette': ['red']}, 'FLOOD DETECTED')
            
            # Stats
            area = flood_final.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), roi, 10).getInfo()
            flooded_ha = area.get('VV', 0) / 10000
            
            # --- DOWNLOAD FEATURE ---
            try:
                download_url = flood_final.getDownloadURL({
                    'name': f'Flood_Mask_{selected_year}_{selected_event_name.split()[0]}',
                    'scale': 30,
                    'region': roi
                })
                st.sidebar.markdown(f"### 📥 Download Results")
                st.sidebar.link_button("Download Flood GeoTIFF", download_url)
            except:
                pass

        else:
            # Optical Logic
            collection = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(roi).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            before = collection.filterDate(dstr(before_start), dstr(before_end)).median().clip(roi)
            after = collection.filterDate(dstr(after_start), dstr(after_end)).median().clip(roi)
            vis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
            m.add_layer(before, vis, 'Before (Optical)')
            m.add_layer(after, vis, 'After (Optical)')

    # Display Map
    m.to_streamlit(height=600)
    
    if sensor_type == "Sentinel-1 (Radar)":
        st.success(f"🛑 Detected Flood Extent: {flooded_ha:.2f} Hectares")

    # --- AI SECTION ---
    st.divider()
    st.subheader("🤖 AI Situation Report")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about this event..."):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        context = f"Event: {selected_event_name} ({selected_year}). Flooded Area: {flooded_ha} ha. User: {prompt}"
        
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": context}, {"role": "user", "content": prompt}])
            reply = response.choices[0].message.content
            st.chat_message("assistant").write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"AI Error: {e}")
