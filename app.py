import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
from openai import OpenAI  # Import OpenAI library

# --- 1. CONFIGURATION & AUTH ---
st.set_page_config(page_title="FDEP SAR Flood Intelligence", layout="wide")

# Trigger EE Authentication
try:
    ee.Initialize(project='flood-intelligence-gee-12345') # REPLACE with your project ID
except:
    ee.Authenticate()
    ee.Initialize(project='flood-intelligence-gee-12345')

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("FDEP Flood Intelligence")
st.sidebar.info("Sentinel-1 SAR Analysis | NISAR-Ready Architecture")

location_name = st.sidebar.text_input("Target Location", "Fort Myers, FL")
# Coordinates for Fort Myers
lat = 26.64
lon = -81.87

# Date Selection
st.sidebar.subheader("Analysis Window")
# Fixed: Using pd.to_datetime explicitly to avoid errors
before_start = st.sidebar.date_input("Before Start", value=pd.to_datetime("2022-09-01"))
before_end = st.sidebar.date_input("Before End", value=pd.to_datetime("2022-09-15"))
after_start = st.sidebar.date_input("After Start", value=pd.to_datetime("2022-09-29"))
after_end = st.sidebar.date_input("After End", value=pd.to_datetime("2022-10-05"))

# --- 3. CORE LOGIC (YOUR SAR ALGORITHM) ---
def get_sar_layer(start, end, roi):
    # NISAR-Ready: Filters for VV/VH polarization
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filterBounds(roi)
                  .filterDate(str(start), str(end)))
    return collection.mosaic().clip(roi)

# Run Analysis when button clicked
if st.sidebar.button("Run Flood Analysis"):
    with st.spinner('Accessing Sentinel-1 Satellite Data...'):
        roi = ee.Geometry.Point([lon, lat]).buffer(20000)
        
        # Load Images
        before = get_sar_layer(before_start, before_end, roi)
        after = get_sar_layer(after_start, after_end, roi)

        # Speckle Filtering
        smooth_radius = 50
        before_filtered = before.focal_mean(smooth_radius, 'circle', 'meters')
        after_filtered = after.focal_mean(smooth_radius, 'circle', 'meters')

        # Change Detection (Ratio)
        difference = after_filtered.divide(before_filtered)
        
        # Thresholding (Flood < 0.8)
        THRESHOLD = 0.8
        flood_mask = difference.select('VV').lt(THRESHOLD)
        
        # Mask out permanent water
        perm_water = before_filtered.select('VV').gt(-15)
        flood_final = flood_mask.updateMask(perm_water).selfMask() 

        # --- 4. VISUALIZATION ---
        st.subheader(f"Flood Impact Analysis: {location_name}")
        
        # Create a Split-Panel Map
        m = geemap.Map(center=[lat, lon], zoom=11)
        m.add_layer(before_filtered, {'min': -25, 'max': 0}, 'Before Storm (SAR)')
        m.add_layer(after_filtered, {'min': -25, 'max': 0}, 'After Storm (SAR)')
        m.add_layer(flood_final, {'palette': ['red']}, 'FLOOD EXTENT')

        m.to_streamlit(height=600)
        
        # --- 5. STATISTICS ---
        # Calculate flooded area in hectares
        flood_area = flood_final.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        # FIXED: Use .get() to prevent crash if no flood is found
        flooded_ha = flood_area.get('flood_detected', 0) / 10000
        
        st.success(f" Detected Flood Extent: {flooded_ha:.2f} Hectares")
        
        # --- 6. AI ASSISTANT IMPLEMENTATION ---
        st.divider()
        st.subheader("🤖 AI Impact Assessment")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask about this flood event..."):
            
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # --- THE "BRAIN" (CONTEXT INJECTION) ---
            system_context = f"""
            You are an expert Flood Response Analyst for the FDEP.
            Current Analysis Context:
            - Location: {location_name}
            - Date Range: {before_start} to {after_end}
            - Detected Flood Extent: {flooded_ha:.2f} hectares.
            
            User Question: {prompt}
            
            Provide a professional, concise assessment. If the flood extent is high (>1000 ha), express urgency.
            """

            try:
                # securely fetch the key from Streamlit Cloud Secrets
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_context},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                ai_response = response.choices[0].message.content
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"AI Error: {e}")
                st.info("API Key missing! You need to add 'OPENAI_API_KEY' to your Streamlit Cloud Secrets.")

else:
    st.write(" Set parameters and click 'Run Flood Analysis' to start.")
