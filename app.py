import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd

st.set_page_config(page_title="FDEP SAR Flood Intelligence", layout="wide")

try:
    ee.Initialize(project='flood-intelligence-gee-12345') # REPLACE with your project ID
except:
    ee.Authenticate()
    ee.Initialize(project='flood-intelligence-gee-12345')

st.sidebar.title("FDEP Flood Intelligence")
st.sidebar.info("Sentinel-1 SAR Analysis | NISAR-Ready Architecture")

location_name = st.sidebar.text_input("Target Location", "Fort Myers, FL")

lat = 26.64
lon = -81.87

st.sidebar.subheader("Analysis Window")

before_start = st.sidebar.date_input("Before Start", value=pd.to_datetime("2022-09-01"))
before_end = st.sidebar.date_input("Before End", value=pd.to_datetime("2022-09-15"))
after_start = st.sidebar.date_input("After Start", value=pd.to_datetime("2022-09-29"))
after_end = st.sidebar.date_input("After End", value=pd.to_datetime("2022-10-05"))


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


        st.subheader(f"Flood Impact Analysis: {location_name}")
        
 
        m = geemap.Map(center=[lat, lon], zoom=11)
        m.add_layer(before_filtered, {'min': -25, 'max': 0}, 'Before Storm (SAR)')
        m.add_layer(after_filtered, {'min': -25, 'max': 0}, 'After Storm (SAR)')
        m.add_layer(flood_final, {'palette': ['red']}, 'FLOOD EXTENT')

        m.to_streamlit(height=600)
        
  
      
        flood_area = flood_final.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        

        flooded_ha = flood_area.get('flood_detected', 0) / 10000
        
        st.success(f" Detected Flood Extent: {flooded_ha:.2f} Hectares")
        st.caption("This statistic will be fed into the LLM Assistant in Phase 3.")

else:
    st.write("Set parameters and click 'Run Flood Analysis' to start.")
