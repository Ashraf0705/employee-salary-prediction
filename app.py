import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the encoders
model = joblib.load("best_model.pkl")

# --- Mappings from your Jupyter Notebook's LabelEncoder ---
workclass_map = {'Federal-gov': 0, 'Local-gov': 1, 'Others': 2, 'Private': 3, 'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6}
marital_status_map = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6}
occupation_map = {'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Other-service': 7, 'Others': 8, 'Priv-house-serv': 9, 'Prof-specialty': 10, 'Protective-serv': 11, 'Sales': 12, 'Tech-support': 13, 'Transport-moving': 14}
relationship_map = {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5}
race_map = {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4}
gender_map = {'Female': 0, 'Male': 1}

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")

# --- Sidebar for User Input ---
st.sidebar.header("Input Employee Details")

# --- Collect all 13 features the model was trained on ---
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Work Class", list(workclass_map.keys()))
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 500000, 150000)
educational_num = st.sidebar.slider("Education Level (Numeric)", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
race = st.sidebar.selectbox("Race", list(race_map.keys()))
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.number_input("Native Country (Numeric code)", 0, 41, 39)

# --- Preprocess inputs and create DataFrame ---
input_data = {
    'age': age,
    'workclass': workclass_map[workclass],
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status_map[marital_status],
    'occupation': occupation_map[occupation],
    'relationship': relationship_map[relationship],
    'race': race_map[race],
    'gender': gender_map[gender],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

# Create a DataFrame (still needed for the model prediction)
input_df = pd.DataFrame([input_data])


# --- Display the input data in two columns ---
st.subheader("Selected Input Details")
col1, col2 = st.columns(2)

features = list(input_data.keys())
values = list(input_data.values())

midpoint = len(features) // 2
with col1:
    for i in range(midpoint):
        label = features[i].replace('_', ' ').replace('-', ' ').title()
        st.markdown(f"**{label}:** `{values[i]}`")

with col2:
    for i in range(midpoint, len(features)):
        label = features[i].replace('_', ' ').replace('-', ' ').title()
        st.markdown(f"**{label}:** `{values[i]}`")


# --- Prediction ---
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    st.subheader("Prediction")
    if prediction[0] == '<=50K':
        st.success(f"The predicted salary is **{prediction[0]}**")
    else:
        st.error(f"The predicted salary is **{prediction[0]}**")
        
    st.subheader("Prediction Probability")
    st.write(f"Probability of earning ≤50K: **{probability[0][0]:.2f}**")
    st.write(f"Probability of earning >50K: **{probability[0][1]:.2f}**")