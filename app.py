import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MPG Predictor", page_icon="⛽")


@st.cache_resource
def load():
    model = joblib.load("mpg_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load()
r = meta["ranges"]

st.title("⛽ Fuel Efficiency (MPG) Predictor")
st.caption(f"Model: {meta['best_model']}, trained on the auto-mpg dataset")

col1, col2 = st.columns(2)
with col1:
    cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
    displacement = st.number_input("Displacement (cu. in.)", 50.0, 600.0, float(r["displacement"][2]))
    horsepower = st.number_input("Horsepower", 40.0, 300.0, float(r["horsepower"][2]))
with col2:
    weight = st.number_input("Weight (lbs)", 1000.0, 6000.0, float(r["weight"][2]))
    acceleration = st.number_input("Acceleration (0-60 mph, sec)", 5.0, 30.0, float(r["acceleration"][2]))
    model_year = st.slider("Model year (19xx)", 70, 82, 76)
origin_label = st.radio("Origin", ["USA", "Europe", "Japan"], horizontal=True)
origin = {"USA": "1", "Europe": "2", "Japan": "3"}[origin_label]

if st.button("Predict MPG", type="primary"):
    row = pd.DataFrame([{
        "cylinders": cylinders, "displacement": displacement, "horsepower": horsepower,
        "weight": weight, "acceleration": acceleration, "model year": model_year, "origin": origin,
    }])
    st.metric("Predicted MPG", f"{model.predict(row)[0]:.1f}")

with st.expander("Model comparison"):
    st.dataframe(pd.DataFrame(meta["results"]).set_index("model"))
