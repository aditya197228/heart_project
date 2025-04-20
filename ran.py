import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from google import genai
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# Your Gemini API Key (Paste here)
# ----------------------------
GEMINI_API_KEY = "AIzaSyDyx5_5Nq99TvvGnU1p_9oeXYlqg2PiISQ"

# ----------------------------
# Data Loading & Preprocessing
# ----------------------------

@st.cache_data(ttl=10800)
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    irrelevant_columns = ["id", "dataset"]
    df = df.drop(columns=[col for col in irrelevant_columns if col in df.columns], errors="ignore")
    df.fillna(df.median(numeric_only=True), inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"Error: Column '{target_col}' not found in the dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns.tolist(), scaler

@st.cache_data(ttl=10800)
def train_model(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "heart_disease_model.pkl")
    return model

# ----------------------------
# ECG Image CNN Training
# ----------------------------

def train_ecg_image_model(image_dir, selected_labels):
    if not os.path.exists(image_dir):
        raise ValueError(f"Dataset directory '{image_dir}' does not exist.")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        image_dir,
        target_size=(180, 180),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        image_dir,
        target_size=(180, 180),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    class_indices = train_data.class_indices
    with open("ecg_class_indices.json", "w") as f:
        json.dump(class_indices, f)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save("ecg_image_cnn_model.h5")
    return model

# ----------------------------
# Prediction and Risk Analysis
# ----------------------------

def assess_risk_factors(user_data):
    recommendations = []
    if user_data["chol"] > 240:
        recommendations.append("Reduce cholesterol by eating more fiber, reducing saturated fats, and exercising regularly.")
    if user_data["trestbps"] > 130:
        recommendations.append("Lower blood pressure through a low-sodium diet, regular exercise, and stress management.")
    if user_data["thalach"] < 100:
        recommendations.append("Increase cardiovascular activity to improve heart rate performance.")
    if not recommendations:
        recommendations.append("Maintain a balanced diet, stay active, and monitor your health regularly.")
    return recommendations

def predict_heart_disease(user_data, model, scaler, feature_names):
    input_array = np.array([user_data[feature] for feature in feature_names]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    risk = "High" if prediction == 1 else "Low"
    recommendations = assess_risk_factors(user_data)
    return "Heart Disease Detected" if prediction == 1 else "No Heart Disease", risk, recommendations

# ----------------------------
# Gemini API Integration
# ----------------------------

def get_gemini_response(query, user_data):
    try:
        context = "\n".join([f"{key}: {value}" for key, value in user_data.items()])
        prompt = f"""
You are a helpful medical assistant. Based on the following user health data, provide medical advice or answer the question below. Keep your tone simple, clear, and positive.

User Health Data:
{context}

User Question:
{query}

Instructions:
- If the user has heart disease, suggest lifestyle and diet changes.
- If not, give health tips to stay safe.
- You can recommend when to see a doctor if needed.
"""
        client = genai.Client(api_key="AIzaSyCqGIPvPHfbKfn8zE7ZW49ReMQPmokLP9g")
        response = client.models.generate_content(
            model="gemini-1.5-pro-latest",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error with Gemini API: {e}"

# ----------------------------
# Streamlit App Layout
# ----------------------------

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("💓 Heart Disease Prediction and Risk Analysis")
st.markdown("---")

file_path = "heart.csv"
X_scaled, y, feature_names, scaler = load_and_preprocess_data(file_path)
model = train_model(X_scaled, y)

st.header("Enter Your Health Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type", options=[("None", 0), ("Mild", 1), ("Moderate", 2), ("Severe", 3)], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar (>120 mg/dL)", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

with col2:
    restecg = st.selectbox("ECG Results", options=[("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)], format_func=lambda x: x[0])[1]
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])[1]
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia Type", options=[("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2)], format_func=lambda x: x[0])[1]

input_data = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}

if st.button("Predict Heart Disease"):
    prediction, risk_level, recommendations = predict_heart_disease(input_data, model, scaler, feature_names)
    st.markdown("---")
    st.header("Prediction Result")
    if prediction == "Heart Disease Detected":
        st.error(f"**Prediction:** {prediction}")
    else:
        st.success(f"**Prediction:** {prediction}")
    st.warning(f"**Risk Level:** {risk_level}")
    st.header("Health Recommendations")
    for rec in recommendations:
        st.write(f"- {rec}")
    st.markdown("---")

# ----------------------------
# ECG Image Prediction
# ----------------------------

st.header("📷 ECG Waveform Image Prediction")
uploaded_image = st.file_uploader("Upload ECG waveform image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded ECG Image", use_column_width=True)

    image = image.resize((180, 180))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    model = tf.keras.models.load_model("ecg_image_cnn_model.h5")
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        st.success(f"Predicted Label: Normal")
    else:
        st.error(f"Predicted Label: Abnormal")

# ----------------------------
# Gemini Assistant with Automatic Response
# ----------------------------

st.header("Personalized Health Assistance")
user_query = st.text_area("Ask a health-related question (e.g., 'How to improve my cholesterol levels?')")

# Automatically trigger the Gemini response when the user types
if user_query.strip():  # Check if there's a valid query
    response = get_gemini_response(user_query, input_data)  # Get response from Gemini API
    st.subheader("Gemini's Response:")
    st.write(response)  # Display the response
else:
    st.warning("Please enter a valid question.")  # Show a warning if no query is entered
