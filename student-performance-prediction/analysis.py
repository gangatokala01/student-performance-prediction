import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 1. Load Dataset
df = pd.read_csv("Dataset.csv")

# 2. Basic Data Analysis
print(df.head())
print(df.info())
print(df.describe())
print(df.corr(numeric_only=True))


# 3. Visualization
plt.scatter(df["PreviousScore"], df["FinalScore"])
plt.xlabel("Previous Score")
plt.ylabel("Final Score")
plt.title("Previous Score vs Final Score")
plt.show()


# 4. Features and Target
X = df[
    ["StudyHours", "Attendance", "PreviousScore",
     "AssignmentsScore", "SleepHours"]
]

y = df["FinalScore"]


# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Create and Train Model
model = LinearRegression()

model.fit(X_train, y_train)


# 7. Test Model
y_pred = model.predict(X_test)

print("Predicted Test Values:", y_pred)


# 8. Model Evaluation
score = r2_score(y_test, y_pred)

print("R2 Score:", score)


# 9. Student Performance Prediction
print("\nStudent Performance Prediction")

study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance: "))
previous_score = float(input("Previous Score: "))
assignments_score = float(input("Assignments Score: "))
sleep_hours = float(input("Sleep Hours: "))


# 10. Create New Student Data
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


# 11. Predict Final Score
prediction = model.predict(new_student)

print("Predicted Final Score:", prediction[0])