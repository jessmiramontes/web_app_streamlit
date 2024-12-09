import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('knn_pipeline_model.joblib')

# Class labels
class_dict = {
    "0": "Insufficient_Weight",
    "1": "Normal_Weight",
    "2": "Obesity_Type_I",
    "3": "Obesity_Type_II",
    "4": "Obesity_Type_III",
    "5": "Overweight_Level_I",
    "6": "Overweight_Level_II"
}

# Define the app
st.title('Obesity Model Prediction')

# Input form
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.number_input('Age', min_value=1.0, step=0.01)
height = st.number_input('Height', step=0.01)
weight = st.number_input('Weight', step=0.01)
family_history_with_overweight = st.selectbox('Family History of Overweight', ['yes', 'no'])
favc = st.selectbox('Do you eat high caloric food frequently?', ['yes', 'no'])
fcvc = st.number_input('How many cups of vegetables do you eat daily?', step=0.01)
ncp = st.number_input('How many main meals do you have daily?', step=0.01)
caec = st.selectbox('>Do you eat any food between meals?', ['Sometimes', 'Frequently', 'Always', 'no'])
smoke = st.selectbox('Do you smoke?', ['yes', 'no'])
ch2o = st.number_input('How much water do you drink daily?', step=0.01)
scc = st.selectbox('Do you monitor the calories you eat daily?', ['yes', 'no'])
faf = st.number_input('How often do you have physical activity?', step=0.01)
tue = st.number_input('How much time do you use technological devices daily?', step=0.01)
calc = st.selectbox('How often do you drink alcohol?', ['Sometimes', 'Frequently', 'Always', 'no'])
mtrans = st.selectbox('Which transportation do you usually use?', ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])

if st.button('Predict'):
    # Create DataFrame
    data = {
        'gender': [gender],
        'age': [age],
        'height': [height],
        'weight': [weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'favc': [favc],
        'fcvc': [fcvc],
        'ncp': [ncp],
        'caec': [caec],
        'smoke': [smoke],
        'ch2o': [ch2o],
        'scc': [scc],
        'faf': [faf],
        'tue': [tue],
        'calc': [calc],
        'mtrans': [mtrans]
    }
    
    input_df = pd.DataFrame(data)

    # Make prediction
    prediction = model.predict(input_df)[0]
    predicted_class = class_dict[str(prediction)]

    st.write(f'The predicted class is: {predicted_class}')
