import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
from openai import OpenAI
import json
import os

# --- 1. CONFIGURATION & AUTH ---
st.set_page_config(page_title="FDEP SAR Flood Intelligence", layout="wide")

def auth_ee():
    try:
        # Trigger authentication flow
        ee.Initialize(project='flood-intelligence-gee-12345')
    except Exception:
        # If standard init fails, try using the saved token from Secrets
        if "EARTHENGINE_TOKEN" in st.secrets:
            # Write the token to the specific path where GEE looks for it
            credentials_path = os.path.expanduser("~/.config/earthengine/")
            os.makedirs(credentials_path, exist_ok=True)
            
            token_content = st.secrets["EARTHENGINE_TOKEN"]
            # Ensure it's a string
            if not isinstance(token_content, str):
                token_content = json.dumps(token_content)
            
            with open(os.path.join(credentials_path, "credentials"), "w") as f:
                f.write(token_content)
            
            # Try initializing again with the new credentials file
            ee.Initialize(project='flood-intelligence-gee-12345')
        else:
            # If no token is found, raise the original error
            raise

# Run the authentication
auth_ee()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("🌊 FDEP Flood Intelligence")
st.sidebar.info("Sentinel-1 SAR Analysis | NISAR-Ready Architecture")

location_name = st.sidebar.text_input("Target Location", "Fort Myers, FL")
lat = 26.64
lon = -81.87

# Date Selection
st.sidebar.subheader("Analysis Window")
before_start = st.sidebar.date_input("Before Start", value=pd.to_datetime("2022-09-01"))
before_end = st.sidebar.date_input("Before End", value=pd.to_datetime("2022-09-15"))
after_start = st.sidebar.date_input("After Start", value=pd.to_datetime("2022-09-29"))
after_end = st.sidebar.date_input("After End", value=pd.to_datetime("2022-10-05"))

# --- 3. CORE LOGIC (SAR ALGORITHM) ---
def get_sar_layer(start, end, roi):
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filterBounds(roi)
                  .filterDate(str(start), str(end)))
    return collection.mosaic().clip(roi)

if st.sidebar.button("Run Flood Analysis"):
    with st.spinner('Accessing Sentinel-1 Satellite Data...'):
        roi = ee.Geometry.Point([lon, lat]).buffer(20000)
        
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

        # --- 4. VISUALIZATION ---
        st.subheader(f"Flood Impact Analysis: {location_name}")
        
        m = geemap.Map(center=[lat, lon], zoom=11)
        m.add_layer(before_filtered, {'min': -25, 'max': 0}, 'Before Storm (SAR)')
        m.add_layer(after_filtered, {'min': -25, 'max': 0}, 'After Storm (SAR)')
        m.add_layer(flood_final, {'palette': ['red']}, 'FLOOD EXTENT')

        m.to_streamlit(height=600)
        
        # --- 5. STATISTICS ---
        flood_area = flood_final.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        flooded_ha = flood_area.get('flood_detected', 0) / 10000
        
        st.success(f"🛑 Detected Flood Extent: {flooded_ha:.2f} Hectares")
        
        # --- 6. AI ASSISTANT ---
        st.divider()
        st.subheader("🤖 AI Impact Assessment")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about this flood event..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            system_context = f"""
            You are an expert Flood Response Analyst.
            Context: {location_name}, {flooded_ha:.2f} ha flooded.
            User Question: {prompt}
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
    st.write("👈 Set parameters and click 'Run Flood Analysis' to start.")
