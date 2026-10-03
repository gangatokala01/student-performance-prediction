import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Prediction")
st.write(
    "Enter the student's academic and lifestyle details "
    "to predict the final score."
)

st.divider()

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Dataset.csv")

# -----------------------------
# Features and Target
# -----------------------------
X = df[
    [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsScore",
        "SleepHours"
    ]
]

y = df["FinalScore"]

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Student Inputs
# -----------------------------
st.subheader("📋 Student Details")

study_hours = st.number_input(
    "📚 Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "📅 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

previous_score = st.number_input(
    "📝 Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

assignments_score = st.number_input(
    "📖 Assignments Score",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

sleep_hours = st.number_input(
    "😴 Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=8.0
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Final Score"):

    new_student = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_score,
            assignments_score,
            sleep_hours
        ]],
        columns=[
            "StudyHours",
            "Attendance",
            "PreviousScore",
            "AssignmentsScore",
            "SleepHours"
        ]
    )

    prediction = model.predict(new_student)[0]

    # Keep score between 0 and 100
    prediction = max(0, min(100, prediction))

    # -------------------------
    # Performance Category
    # -------------------------
    if prediction >= 90:
        category = "Excellent 🌟"
    elif prediction >= 75:
        category = "Good 👍"
    elif prediction >= 60:
        category = "Average 🙂"
    else:
        category = "Needs Improvement 📚"

    # -------------------------
    # Display Result
    # -------------------------
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Predicted Final Score",
        value=f"{prediction:.2f}"
    )

    st.success(
        f"Performance Level: **{category}**"
    )

    st.progress(int(prediction))

st.divider()

st.caption(
    "AI-Powered Student Performance Prediction & Analytics System"
)