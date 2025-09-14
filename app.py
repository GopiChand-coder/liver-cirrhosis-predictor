# app.py
from flask import Flask, request, jsonify, render_template
import joblib, numpy as np, math, datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load trained model (ensure liver_model.pkl exists in project root)
model = joblib.load("liver_model.pkl")

# --- Database (SQLite) via SQLAlchemy ---
DATABASE_URL = "sqlite:///patients.db"
engine = create_engine(DATABASE_URL, echo=False, future=True)
Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # store a subset/full set as needed; keep columns to match earlier design
    age = Column(Float)
    gender = Column(Integer)
    bilirubin = Column(Float)
    albumin = Column(Float)
    ast = Column(Float)
    alt = Column(Float)
    alp = Column(Float)
    ggt = Column(Float)
    inr = Column(Float)
    platelets = Column(Float)
    hemoglobin = Column(Float)
    wbc = Column(Float)
    prothrombin = Column(Float)
    creatinine = Column(Float)
    sodium = Column(Float)
    glucose = Column(Float)
    bun = Column(Float)
    bmi = Column(Float)
    bp_systolic = Column(Float)
    bp_diastolic = Column(Float)
    ferritin = Column(Float)
    vitd = Column(Float)
    cholesterol = Column(Float)
    alcohol = Column(Float)
    ascites = Column(Integer)
    encephalopathy = Column(Integer)
    prediction = Column(String(50))
    prob_high = Column(Float, nullable=True)

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False)

# helper: safe float conversion
def safe_float(v, default=0.0):
    try:
        if v is None or v == "" or str(v).lower() == "null":
            return float(default)
        return float(v)
    except Exception:
        return float(default)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept JSON or form-data (fallback)
        if request.is_json:
            data = request.get_json()
        else:
            # request.form is an ImmutableMultiDict
            data = request.form.to_dict()

        if not isinstance(data, dict):
            data = {}

        # Required fields (basic)
        required = ["age", "bilirubin", "albumin", "inr", "creatinine"]
        for r in required:
            if r not in data or data.get(r) in (None, "", "null"):
                return jsonify({"error": f"Missing required field: {r}"}), 400

        # Define feature order expected by model (use subset present in form)
        keys = ["age","gender","bilirubin","albumin","ast","alt","alp","ggt",
                "inr","platelets","hemoglobin","wbc","prothrombin","creatinine","sodium",
                "glucose","bun","bmi","bp_systolic","bp_diastolic","ferritin","vitd","cholesterol","alcohol",
                "ascites","encephalopathy"]

        features = [ safe_float(data.get(k)) for k in keys ]
        arr = np.array([features])

        # Make prediction
        pred_class = model.predict(arr)[0]
        try:
            pred_class = int(pred_class)
        except:
            pred_class = int(np.round(float(pred_class)))

        # Get probability for class '1' (High Risk), robustly
        prob_high = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(arr)[0]
                classes = list(model.classes_)
                if 1 in classes:
                    idx = classes.index(1)
                    prob_high = float(proba[idx])
                else:
                    # fallback to last column
                    prob_high = float(proba[-1])
            except Exception:
                prob_high = None

        result_text = "High Risk" if pred_class == 1 else "Low Risk"

        # Child-Pugh calculation (simplified)
        bilirubin = safe_float(data.get("bilirubin"))
        albumin = safe_float(data.get("albumin"))
        inr_val = safe_float(data.get("inr"))
        ascites = int(safe_float(data.get("ascites", 0)))
        enceph = int(safe_float(data.get("encephalopathy", 0)))

        cp_pts = 0
        cp_pts += 1 if bilirubin < 2 else 2 if bilirubin <= 3 else 3
        cp_pts += 1 if albumin > 3.5 else 2 if albumin >= 2.8 else 3
        cp_pts += 1 if inr_val < 1.7 else 2 if inr_val <= 2.3 else 3
        cp_pts += 1 if ascites == 0 else 3
        cp_pts += 1 if enceph == 0 else 3
        if cp_pts <= 6:
            cp_class = "Child-Pugh A (Mild)"
        elif cp_pts <= 9:
            cp_class = "Child-Pugh B (Moderate)"
        else:
            cp_class = "Child-Pugh C (Severe)"

        # MELD (log-based) with safe minimums
        try:
            bil_min = max(bilirubin, 1.0)
            cre_min = max(safe_float(data.get("creatinine", 1.0)), 1.0)
            inr_min = max(inr_val, 1.0)
            meld = 3.78 * math.log(bil_min) + 11.2 * math.log(inr_min) + 9.57 * math.log(cre_min) + 6.43
            meld = round(meld, 2)
        except Exception:
            meld = None

        # persist into DB (safe defaults)
        session = SessionLocal()
        patient = Patient(
            age = safe_float(data.get("age")),
            gender = int(safe_float(data.get("gender", 0))),
            bilirubin = bilirubin,
            albumin = albumin,
            ast = safe_float(data.get("ast")),
            alt = safe_float(data.get("alt")),
            alp = safe_float(data.get("alp")),
            ggt = safe_float(data.get("ggt")),
            inr = inr_val,
            platelets = safe_float(data.get("platelets")),
            hemoglobin = safe_float(data.get("hemoglobin")),
            wbc = safe_float(data.get("wbc")),
            prothrombin = safe_float(data.get("prothrombin")),
            creatinine = safe_float(data.get("creatinine")),
            sodium = safe_float(data.get("sodium")),
            glucose = safe_float(data.get("glucose")),
            bun = safe_float(data.get("bun")),
            bmi = safe_float(data.get("bmi")),
            bp_systolic = safe_float(data.get("bp_systolic")),
            bp_diastolic = safe_float(data.get("bp_diastolic")),
            ferritin = safe_float(data.get("ferritin")),
            vitd = safe_float(data.get("vitd")),
            cholesterol = safe_float(data.get("cholesterol")),
            alcohol = safe_float(data.get("alcohol")),
            ascites = ascites,
            encephalopathy = enceph,
            prediction = result_text,
            prob_high = prob_high
        )
        session.add(patient)
        session.commit()
        session.close()

        return jsonify({
            "prediction": result_text,
            "probability_high_risk": prob_high,
            "child_pugh": cp_class,
            "meld_score": meld
        })
    except Exception as e:
        # Always return JSON on exceptions (no HTML tracebacks)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/history", methods=["GET"])
def history():
    try:
        session = SessionLocal()
        rows = session.query(Patient).order_by(Patient.created_at.desc()).limit(20).all()
        session.close()
        rows_ser = []
        for r in rows:
            rows_ser.append({
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "age": r.age,
                "prediction": r.prediction,
                "prob_high": r.prob_high
            })
        return jsonify({"rows": rows_ser})
    except Exception as e:
        return jsonify({"error": f"History load failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
