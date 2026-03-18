import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Smart Gym Diet Planner",
    page_icon="🏋",
    layout="wide"
)

# ---------------------------------------------------
# STYLING (Black Text & Optimized Font Sizes)
# ---------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

/* GLOBAL SETTINGS */
html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif;
    color: black !important;
    background-color: #ffe6f0;
}

/* TITLE & HEADERS */
.main-title { 
    font-size: 48px; 
    font-weight: 800; 
    text-align: center; 
    color: black; 
    margin-bottom: 0px;
}
.sub-title { 
    font-size: 24px; 
    text-align: center; 
    color: black; 
    margin-top: 0px;
}
.section-title { 
    font-size: 22px; 
    font-weight: 700; 
    color: black; 
    padding-top: 20px;
}

/* Target Subheaders and Subtitles */
h1, h2, h3, .stMarkdown h3 {
    color: black !important;
    font-size: 20px !important;
}

/* INPUT LABELS */
label {
    color: black !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* BUTTON */
.stButton>button {
    background-color: #ff2e63;
    color: white !important;
    font-size: 16px;
    padding: 8px 25px;
    border-radius: 10px;
    border: none;
}

/* TABS STYLE */
.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
    justify-content: center;
}

.stTabs [data-baseweb="tab"] {
    background-color: #ffffff;
    padding: 10px 25px;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
    color: black !important;
    border: 1px solid #ddd;
}

.stTabs [aria-selected="true"] {
    background-color: #ff2e63;
    color: white !important;
}

/* METRICS - Fixes White Text and Large Size */
[data-testid="stMetricValue"] {
    color: black !important;
    font-size: 22px !important;
}

[data-testid="stMetricLabel"] {
    color: black !important;
    font-size: 14px !important;
}

/* SMALL TEXT & DIET HEADINGS */
.small-text, .diet-heading, .diet-text {
    font-size: 16px;
    color: black;
    font-weight: 600;
}

/* TABLE TEXT */
[data-testid="stDataFrame"] div[role="gridcell"] {
    font-size: 13px !important;
    color: black !important;
}

/* Fix for generic text elements */
.stMarkdown p, .stText {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown('<p class="main-title">Smart Gym Diet Planner</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Diet + Food Analyzer</p>', unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA (Ensure food.csv exists in your directory)
# ---------------------------------------------------

try:
    food_df = pd.read_csv("food.csv")
except FileNotFoundError:
    st.error("Error: 'food.csv' not found. Please ensure the file is in the app directory.")
    st.stop()

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "👤 User Planner",
    "🥗 Diet Plan",
    "📷 Food Analyzer"
])

# ===================================================
# TAB 1: User Planner
# ===================================================

with tab1:
    st.markdown('<p class="section-title">Enter Your Details</p>', unsafe_allow_html=True)
    with st.form("form"):
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("Age", 10, 80, value=25)
            height = st.number_input("Height (cm)", 120, 230, value=170)
            weight = st.number_input("Weight (kg)", 30, 200, value=70)
        with col_b:
            gender = st.selectbox("Gender", ["Male", "Female"])
            activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
            goal = st.selectbox("Goal", ["Weight Loss", "Maintenance", "Weight Gain"])

        submit = st.form_submit_button("Calculate My Plan")

    if submit:
        if gender == "Male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        if activity == "Low":
            calories = bmr * 1.2
        elif activity == "Moderate":
            calories = bmr * 1.55
        else:
            calories = bmr * 1.9

        st.session_state.calories = int(calories)
        st.session_state.goal = goal

        st.success(f"🔥 Daily Calories: {int(calories)} kcal")
        st.info(f"🎯 Goal: {goal}")

# ===================================================
# TAB 2: Diet Plan
# ===================================================

with tab2:
    st.markdown('<p class="section-title">Your Personalized Diet Plan</p>', unsafe_allow_html=True)

    if "calories" not in st.session_state:
        st.warning("⚠ Please fill Tab 1 first to calculate your needs.")
    else:
        calories = st.session_state.calories
        goal = st.session_state.goal

        if goal == "Weight Loss":
            result = food_df.sort_values(by="Calories").head(6)
        elif goal == "Weight Gain":
            result = food_df.sort_values(by="Calories", ascending=False).head(6)
        else:
            result = food_df.sample(6)

        result = result.reset_index(drop=True)
        result.index = result.index + 1
        result["Grams"] = ((calories / result["Calories"]) * 100).round(0)

        st.dataframe(result, use_container_width=True)

        st.markdown('<p class="diet-heading">🍽 Suggested Daily Portions</p>', unsafe_allow_html=True)

        breakfast = result.iloc[:2]
        lunch = result.iloc[2:4]
        dinner = result.iloc[4:6]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<p class="diet-text">🍳 Breakfast</p>', unsafe_allow_html=True)
            st.write(breakfast[["Food", "Grams"]])
        with col2:
            st.markdown('<p class="diet-text">🍛 Lunch</p>', unsafe_allow_html=True)
            st.write(lunch[["Food", "Grams"]])
        with col3:
            st.markdown('<p class="diet-text">🍲 Dinner</p>', unsafe_allow_html=True)
            st.write(dinner[["Food", "Grams"]])

# ===================================================
# TAB 3: Food Analyzer
# ===================================================

with tab3:
    st.markdown('<p class="section-title">AI Food Analyzer</p>', unsafe_allow_html=True)
    file = st.file_uploader("Upload Food Image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, width=300)

        # Cache the model to avoid reloading on every interaction
        @st.cache_resource
        def load_model():
            return tf.keras.applications.MobileNetV2(weights="imagenet")

        model = load_model()

        img_prep = img.resize((224, 224))
        arr = image.img_to_array(img_prep)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        pred = model.predict(arr)
        decoded = decode_predictions(pred, top=1)[0]
        label = decoded[0][1]
        conf = decoded[0][2]

        st.markdown(f'<p class="small-text">Detected: <b>{label.replace("_", " ").title()}</b></p>', unsafe_allow_html=True)
        st.markdown(f'<p class="small-text">Confidence: {round(conf*100, 2)}%</p>', unsafe_allow_html=True)

        # Matching label to CSV data
        match = None
        for f in food_df["Food"]:
            if f.lower() in label.lower() or label.lower() in f.lower():
                match = f
                break

        if match:
            row = food_df[food_df["Food"] == match]
            st.subheader("🥗 Nutrition (per 100g)")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Calories", f"{row['Calories'].values[0]} kcal")
            c2.metric("Protein", f"{row['Protein'].values[0]}g")
            c3.metric("Fiber", f"{row['Fiber'].values[0]}g")

            c4, c5 = st.columns(2)
            c4.metric("Fat", f"{row['Fat'].values[0]}g")
            c5.metric("Carbs", f"{row['Carbs'].values[0]}g")
        else:
            st.warning("AI detected the food, but nutrition data for this item is not in our database.")