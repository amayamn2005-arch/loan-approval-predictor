
import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "loan_data.csv")

COLUMNS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Gender", "Married",
    "Dependents", "Education", "Self_Employed", "Property_Area",
    "Loan_Status"
]


def create_sample_dataset():
    rows = []

    for i in range(240):
        income = 15000 + (i * 1379) % 135000
        co_income = (i * 853) % 65000
        amount = 30 + (i * 17) % 320
        credit = 1 if i % 5 != 0 else 0

        approved = credit == 1 and income + co_income >= amount * 1000

        rows.append({
            "ApplicantIncome": income,
            "CoapplicantIncome": co_income,
            "LoanAmount": amount,
            "Loan_Amount_Term": 360 if i % 5 else 180,
            "Credit_History": credit,
            "Gender": "Male" if i % 3 else "Female",
            "Married": "Yes" if i % 4 else "No",
            "Dependents": str(i % 4) if i % 7 else "3+",
            "Education": "Graduate" if i % 4 else "Not Graduate",
            "Self_Employed": "Yes" if i % 6 == 0 else "No",
            "Property_Area": ["Urban", "Semiurban", "Rural"][i % 3],
            "Loan_Status": "Y" if approved else "N"
        })

    pd.DataFrame(rows, columns=COLUMNS).to_csv(DATA_FILE, index=False)


def load_data():
    try:
        if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
            raise ValueError("Dataset missing or empty")

        df = pd.read_csv(DATA_FILE)

        if df.empty or not set(COLUMNS).issubset(df.columns):
            raise ValueError("Dataset columns are invalid")

        return df

    except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        print("Creating a fresh demonstration dataset...")
        create_sample_dataset()
        return pd.read_csv(DATA_FILE)


data = load_data()

numeric_features = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]

categorical_features = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]

X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"].map({"Y": 1, "N": 0})

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        applicant = pd.DataFrame([{
            "ApplicantIncome": float(request.form["ApplicantIncome"]),
            "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
            "LoanAmount": float(request.form["LoanAmount"]),
            "Loan_Amount_Term": int(request.form["Loan_Amount_Term"]),
            "Credit_History": int(request.form["Credit_History"]),
            "Gender": request.form["Gender"],
            "Married": request.form["Married"],
            "Dependents": request.form["Dependents"],
            "Education": request.form["Education"],
            "Self_Employed": request.form["Self_Employed"],
            "Property_Area": request.form["Property_Area"]
        }])

        if (
            applicant["ApplicantIncome"].iloc[0] < 0
            or applicant["CoapplicantIncome"].iloc[0] < 0
            or applicant["LoanAmount"].iloc[0] <= 0
            or applicant["Loan_Amount_Term"].iloc[0] <= 0
            or applicant["Credit_History"].iloc[0] not in (0, 1)
        ):
            raise ValueError("Please enter valid income, loan and credit details.")

        prediction = int(model.predict(applicant)[0])
        probabilities = model.predict_proba(applicant)[0]
        confidence = round(float(probabilities[prediction]) * 100, 1)

        result = {
            "approved": prediction == 1,
            "confidence": confidence,
            "loan_amount": applicant["LoanAmount"].iloc[0],
            "income": applicant["ApplicantIncome"].iloc[0]
        }

        return render_template("index.html", result=result, error=None)

    except (ValueError, TypeError, KeyError) as exc:
        return render_template(
            "index.html",
            result=None,
            error=f"Please check your entries: {exc}"
        )


if __name__ == "__main__":
    print("LoanWise Loan Approval Predictor is starting...")
    app.run(debug=True)
