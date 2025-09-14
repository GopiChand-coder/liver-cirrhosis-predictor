# ml_model.py — synthetic training to produce a working liver_model.pkl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Seed for reproducibility
rng = np.random.RandomState(42)
N = 1200

# Create synthetic distributions loosely inspired by plausible ranges
age = rng.randint(18, 90, size=N)
gender = rng.randint(0,2,size=N)
bilirubin = np.round(np.exp(rng.normal(np.log(1.0), 1.0, size=N)), 2)  # some >1
albumin = np.round(rng.normal(3.5, 0.6, size=N),2)
ast = np.round(np.abs(rng.normal(60, 40, size=N)),0)
alt = np.round(np.abs(rng.normal(55, 35, size=N)),0)
alp = np.round(np.abs(rng.normal(110, 80, size=N)),0)
ggt = np.round(np.abs(rng.normal(70, 60, size=N)),0)
inr = np.round(np.clip(rng.normal(1.2, 0.4, size=N), 0.8, 4.0),2)
platelets = np.round(np.clip(rng.normal(220, 90, size=N), 20, 500),0)
hemoglobin = np.round(np.clip(rng.normal(13.5, 2.0, size=N),6,18),1)
wbc = np.round(np.clip(rng.normal(7, 2.5, size=N),1,30),1)
prothrombin = np.round(np.clip(rng.normal(13, 3, size=N),8,40),1)
creatinine = np.round(np.clip(rng.normal(1.0, 0.4, size=N),0.4,5.0),2)
sodium = np.round(np.clip(rng.normal(137, 4, size=N),120,160),1)
glucose = np.round(np.clip(rng.normal(100, 25, size=N),60,400),1)
bun = np.round(np.clip(rng.normal(14,7,size=N),4,80),1)
bmi = np.round(np.clip(rng.normal(26,5,size=N),15,50),1)
bp_systolic = np.round(np.clip(rng.normal(125,15,size=N),80,220),0)
bp_diastolic = np.round(np.clip(rng.normal(78,10,size=N),40,140),0)
ferritin = np.round(np.clip(rng.normal(150,120,size=N),15,2000),1)
vitd = np.round(np.clip(rng.normal(24,10,size=N),5,60),1)
cholesterol = np.round(np.clip(rng.normal(170,50,size=N),90,350),1)
alcohol = np.round(np.clip(rng.exponential(1.0,size=N)*4,0,200),1)
ascites = rng.binomial(1, 0.12, size=N)
encephalopathy = rng.binomial(1, 0.08, size=N)

# Create a synthetic risk label correlated with high bilirubin/low albumin/high INR/ascites
risk_score = (np.log1p(bilirubin) * 1.6) + ((2.8 - albumin).clip(min=0) * 1.2) + (inr * 1.2) + (ascites * 1.5) + (encephalopathy * 1.5)
prob = 1/(1+np.exp(- (risk_score - 2.5)))  # sigmoid
y = (prob > 0.5).astype(int)

df = pd.DataFrame({
    "age": age, "gender": gender, "bilirubin": bilirubin, "albumin": albumin, "ast": ast,
    "alt": alt, "alp": alp, "ggt": ggt, "inr": inr, "platelets": platelets,
    "hemoglobin": hemoglobin, "wbc": wbc, "prothrombin": prothrombin, "creatinine": creatinine,
    "sodium": sodium, "glucose": glucose, "bun": bun, "bmi": bmi, "bp_systolic": bp_systolic,
    "bp_diastolic": bp_diastolic, "ferritin": ferritin, "vitd": vitd, "cholesterol": cholesterol,
    "alcohol": alcohol, "ascites": ascites, "encephalopathy": encephalopathy
})
df['class'] = y

# Train
X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

print("Train acc:", clf.score(X_train, y_train), "Test acc:", clf.score(X_test, y_test))
joblib.dump(clf, "liver_model.pkl")
print("Saved model -> liver_model.pkl")
