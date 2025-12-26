import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
from openai import OpenAI
from geopy.geocoders import Nominatim
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

# --- 2. THE EVENT LIBRARY ---
# This dictionary holds the data for your presets
flood_events = {
    "Custom Location (Manual Input)": {"lat": 0, "lon": 0, "dates": []},
    "Hurricane Ian (Fort Myers, FL)": {
        "lat": 26.64, "lon": -81.87, 
        "dates": ["2022-09-01", "2022-09-15", "2022-09-29", "2022-10-05"]
    },
    "Pakistan Floods (Sindh)": {
        "lat": 26.90, "lon": 68.10, 
        "dates": ["2022-06-01", "2022-06-30", "2022-08-20", "2022-08-30"]
    },
    "Libya Dam Collapse (Derna)": {
        "lat": 32.76, "lon": 22.63, 
        "dates": ["2023-08-01", "2023-09-01", "2023-09-12", "2023-09-20"]
    },
    "California Atmospheric River (Sacramento)": {
        "lat": 38.58, "lon": -121.49,
        "dates": ["2022-12-01", "2022-12-25", "2023-01-05", "2023-01-15"]
    }
}

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🌍 Global Flood Intelligence")

# A. EVENT SELECTOR
selected_event = st.sidebar.selectbox("Select a Historical Event", list(flood_events.keys()))

if selected_event != "Custom Location (Manual Input)":
    # Auto-fill params from dictionary
    params = flood_events[selected_event]
    lat = params["lat"]
    lon = params["lon"]
    d = [pd.to_datetime(x) for x in params["dates"]]
    
    # We still show the search bar but disable it or just show the name
    st.sidebar.info(f"📍 Loaded: {selected_event}")
    location_query = selected_event
    
    # Auto-set dates
    before_start, before_end, after_start, after_end = d[0], d[1], d[2], d[3]

else:
    # Manual Mode
    location_query = st.sidebar.text_input("Enter Location (City, Country)", "New Orleans, USA")
    geolocator = Nominatim(user_agent="flood_app")
    location = geolocator.geocode(location_query)
    if location:
        lat, lon = location.latitude, location.longitude
        st.sidebar.success(f"📍 Found: {location.address}")
    else:
        st.sidebar.error("Location not found.")
        lat, lon = 26.64, -81.87
        
    st.sidebar.subheader("Date Selection")
    before_start = st.sidebar.date_input("Before Start", pd.to_datetime("2022-09-01"))
    before_end = st.sidebar.date_input("Before End", pd.to_datetime("2022-09-15"))
    after_start = st.sidebar.date_input("After Start", pd.to_datetime("2022-09-29"))
    after_end = st.sidebar.date_input("After End", pd.to_datetime("2022-10-05"))

# B. SATELLITE SELECTOR
sensor_type = st.sidebar.radio("Select Satellite Sensor", ["Sentinel-1 (Radar)", "Sentinel-2 (Optical)"])

# --- 4. EXECUTION ---
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

if st.sidebar.button("Run Global Analysis"):
    st.session_state.analysis_active = True

if st.session_state.analysis_active:
    with st.spinner(f'Processing {sensor_type} data for {location_query}...'):
        roi = ee.Geometry.Point([lon, lat]).buffer(20000)
        m = geemap.Map(center=[lat, lon], zoom=11)
        flooded_ha = 0
        
        # --- LOGIC FOR SENTINEL-1 (RADAR) ---
        if sensor_type == "Sentinel-1 (Radar)":
            def get_sar(start, end):
                return (ee.ImageCollection('COPERNICUS/S1_GRD')
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))
                        .filterBounds(roi)
                        .filterDate(str(start), str(end))
                        .mosaic().clip(roi))

            before = get_sar(before_start, before_end)
            after = get_sar(after_start, after_end)

            # Process
            before_smooth = before.focal_mean(50, 'circle', 'meters')
            after_smooth = after.focal_mean(50, 'circle', 'meters')
            diff = after_smooth.divide(before_smooth)
            
            # Flood Mask
            threshold = 0.8
            flood_mask = diff.select('VV').lt(threshold)
            perm_water = before_smooth.select('VV').gt(-15)
            flood_final = flood_mask.updateMask(perm_water).selfMask()
            
            # Add to Map
            m.add_layer(before_smooth, {'min': -25, 'max': 0}, 'Before (SAR)')
            m.add_layer(after_smooth, {'min': -25, 'max': 0}, 'After (SAR)')
            m.add_layer(flood_final, {'palette': ['red']}, 'FLOOD DETECTED')
            
            # Stats
            area = flood_final.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e9
            ).getInfo()
            flooded_ha = area.get('VV', 0) / 10000

        # --- LOGIC FOR SENTINEL-2 (OPTICAL) ---
        else:
            def get_optical(start, end):
                return (ee.ImageCollection('COPERNICUS/S2_SR')
                        .filterBounds(roi)
                        .filterDate(str(start), str(end))
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        .median().clip(roi)) # Use median to remove clouds
            
            before_opt = get_optical(before_start, before_end)
            after_opt = get_optical(after_start, after_end)
            
            vis_params = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']} # True Color
            m.add_layer(before_opt, vis_params, 'Before (Optical)')
            m.add_layer(after_opt, vis_params, 'After (Optical)')
            
            st.warning("⚠️ Note: Optical Flood Detection is visual only. Use Radar for accurate calculations.")
            flooded_ha = 0 # We don't calculate area for optical yet

    # --- 5. VISUALIZATION ---
    st.subheader(f"Analysis: {location_query} | Sensor: {sensor_type}")
    m.to_streamlit(height=600)
    
    if sensor_type == "Sentinel-1 (Radar)":
        st.success(f"🛑 Detected Flood Extent: {flooded_ha:.2f} Hectares")
    
    # --- 6. AI ASSISTANT ---
    st.divider()
    st.subheader("🤖 AI Intelligence")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask for details..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        system_context = f"""
        You are a Global Disaster Response Specialist.
        Current Event: {selected_event}
        Location: {lat}, {lon}
        Sensor Data: {flooded_ha:.2f} ha flooded (if 0, data is optical/visual only).
        
        User Question: {prompt}
        
        Task: 
        1. If the user asks for locations, use your internal knowledge of the coordinates ({lat}, {lon}) to name nearby cities, rivers, or landmarks.
        2. Provide a strategic assessment of the situation.
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
    st.write("👈 Select an Event or enter a custom location.")
