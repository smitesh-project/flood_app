import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the trained model and scaler
loaded_model = pickle.load(open('xgboost.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

def predict_flood(input_data):
    input_df = pd.DataFrame([input_data])
    normalized_input = scaler.transform(input_df)
    prediction = loaded_model.predict(normalized_input)
    return 'Flood' if prediction[0] == 1 else 'No Flood'

def main():
    # Page title
    st.title('Flood Prediction Web App')

    # Input fields with real-life values hints
    st.sidebar.header('Input Features')

    MonsoonIntensity = st.number_input('Monsoon Intensity (0 - 300)', min_value=0.0, max_value=300.0, value=0.0)
    IneffectiveDisasterPreparedness = st.number_input('Ineffective Disaster Preparedness (0 - 10)', min_value=0.0,
                                                      max_value=10.0, value=0.0)
    DeterioratingInfrastructure = st.number_input('Deteriorating Infrastructure (0 - 10)', min_value=0.0,
                                                  max_value=10.0, value=0.0)
    PopulationScore = st.number_input('Population Score (0 - 100)', min_value=0.0, max_value=100.0, value=0.0)
    Siltation = st.number_input('Siltation (0 - 10)', min_value=0.0, max_value=10.0, value=0.0)
    WetlandLoss = st.number_input('Wetland Loss (0 - 500)', min_value=0.0, max_value=500.0, value=0.0)
    PoliticalFactors = st.number_input('Political Factors (0 - 5)', min_value=0.0, max_value=5.0, value=0.0)
    TopographyDrainage = st.number_input('Topography Drainage (0 - 10)', min_value=0.0, max_value=10.0, value=0.0)
    Landslides = st.number_input('Landslides (0 - 50)', min_value=0.0, max_value=50.0, value=0.0)
    DrainageSystems = st.number_input('Drainage Systems (0 - 10)', min_value=0.0, max_value=10.0, value=0.0)

    # Prediction button

    # Predict button
    if st.sidebar.button('Predict'):
        input_data = {
            'MonsoonIntensity': MonsoonIntensity,
            'IneffectiveDisasterPreparedness': IneffectiveDisasterPreparedness,
            'DeterioratingInfrastructure': DeterioratingInfrastructure,
            'PopulationScore': PopulationScore,
            'Siltation': Siltation,
            'WetlandLoss': WetlandLoss,
            'PoliticalFactors': PoliticalFactors,
            'TopographyDrainage': TopographyDrainage,
            'Landslides': Landslides,
            'DrainageSystems': DrainageSystems
        }

        result = predict_flood(input_data)
        st.write(f"Prediction: {result}")

        if result == 'Flood':
            st.warning('Warning: Flood risk detected!')
        else:
            st.success('No Flood risk detected.')

if __name__ == '__main__':
    main()
