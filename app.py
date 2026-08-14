import pandas as pd
import streamlit as st


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Disease Prediction", page_icon="🏥")

st.header("🏥 Disease Prediction System")
st.caption("Predict possible conditions based on symptoms")

# ---------------- LOAD & MODEL ---------------- #
@st.cache_resource
def load_model():
    data = pd.read_csv("disease.csv")

    # Clean
    data = data.drop_duplicates().dropna()

    # 🔥 LABEL GROUPING (IMPORTANT)
    def simplify(d):
        d = d.lower()

        if "flu" in d or "viral" in d or "covid" in d:
            return "Viral Infection"

        elif "bronchitis" in d or "sinusitis" in d or "cold" in d:
            return "Respiratory Issue"

        elif "dengue" in d or "malaria" in d or "typhoid" in d:
            return "Serious Infection"

        elif "food" in d or "gastr" in d or "stomach" in d:
            return "Stomach Issue"

        elif "migraine" in d or "headache" in d:
            return "Headache"
        
        elif "nausea" in d or "vomiting" in d or "diarrhea" in d:
             return "Digestive Issue"

        elif "fatigue" in d:
            return "Fatigue"

        else:
            return "General Condition"

    data["disease"] = data["disease"].apply(simplify)

    # Remove conflicting patterns
    data = data.groupby(list(data.columns[:-1])).agg(lambda x: x.mode()[0]).reset_index()

    X = data.drop("disease", axis=1)
    y = data["disease"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Stratified split (stable accuracy)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Tuned model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, le, X.columns, X_test, y_test

model, le, feature_names, X_test, y_test = load_model()

# ---------------- PERFORMANCE ---------------- #
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.title("📊 Model Performance")
st.sidebar.write(f"Accuracy: {round(accuracy*100,2)}%")

# ---------------- UI ---------------- #
st.subheader("Select Symptoms")

selected_symptoms = st.multiselect(
    "Choose symptoms",
    options=feature_names
)

# Input vector
input_data = pd.DataFrame([[0]*len(feature_names)], columns=feature_names)

for symptom in selected_symptoms:
    input_data[symptom] = 1

st.subheader("Selected Symptoms")
st.dataframe(input_data, hide_index=True)

# ---------------- PREDICTION ---------------- #
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.error("⚠ Please select at least one symptom")

    else:
        if len(selected_symptoms) == 1:
            st.info("⚠ One symptom may not be enough for accurate prediction")

        probs = model.predict_proba(input_data)[0]
        classes = le.inverse_transform(range(len(probs)))

        top = probs.argsort()[-3:][::-1]

        st.subheader("🧠 Most Likely Conditions")

        for i in top:
            st.write(f"👉 {classes[i]} ({round(probs[i]*100,2)}%)")

        st.warning("⚠ This is not a medical diagnosis. Consult a doctor.")

# ---------------- FEATURE IMPORTANCE ---------------- #
st.subheader("📊 Important Symptoms (Top 10)")

importance = model.feature_importances_
features = pd.Series(importance, index=feature_names)

top_features = features.sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(7, 4))
top_features.sort_values().plot(kind='barh', ax=ax)

ax.set_xlabel("Importance")
ax.set_ylabel("Symptoms")

plt.tight_layout()
st.pyplot(fig)

# ---------------- FOOTER ---------------- #
st.divider()
st.caption("👨‍💻 Developed by Sumit Mahato | B.Tech CSE")
