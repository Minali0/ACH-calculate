import pandas as pd
import numpy as np
from tensorflow import keras
import joblib
import tensorflow as tf


def load_model(model_path):
    # Load the model
    model = keras.models.load_model(model_path)
    return model

def load_scaler(scaler_path):
    # Load the scaler
    scaler = joblib.load(scaler_path)
    return scaler


def preprocess_data(user_data):

    user_df = pd.DataFrame({
        'Scenario': [user_data['Scenario']],
        'Particle_size': [user_data['Particle_size']],
        'Position_of_Freestanding_lights': [user_data['Position_of_Freestanding_lights']],
        'Position_of_outlet': [user_data['Position_of_outlet']],
        'Temp': [user_data['Temp']]
    })
   
    # Map categorical variables to numerical values as done in preprocess_data
    user_df['Position_of_Freestanding_lights'] = user_df['Position_of_Freestanding_lights'].map({
        "head":1,
        "torso":2,
        "feet":3,
        "head and feet": 4,
        "head and torso": 5,
        "torso and feet": 6,
        "head and torso and feet": 7,
        
    }).fillna(-1)
    
    user_df['Particle_size'] = user_df['Particle_size'].map({
        '0.7': 0.7,
        '0.5': 0.5,
        '0.4': 0.4,
        '0.1': 0.1,
        '0.045': 0.045,
        '1': 1.0,
        '1.5': 1.5,
        '2.5': 2.5,
        '2': 2.0,
        '3': 3.0,
    }).fillna(-1)
    
            
    user_df['Scenario'] = user_df['Scenario'].map({
       "6 inches below OR table": 1,
        "at the OR table": 2,
        "6 inches above": 3,
        "1 ft above": 4,
        "y-mid point":5,
        "x-mid point":6,
    }).fillna(-1)

    user_df['Position_of_outlet'] = user_df['Position_of_outlet'].map({'a': 1, 'b': 2, 'c': 3}).fillna(-1)

    return user_df


def predict_ach_sum_of_ppm(model, user_df, scaler):
    # Preprocess input data and make a prediction
    input_data_scaled = scaler.transform(user_df)
    predictions = model.predict(input_data_scaled)
    return predictions


def main():

    # Load the model and scaler
    model = load_model("/home/seaflux/Documents/cfd_code_6_sept/model.keras")
    scaler = load_scaler("/home/seaflux/Documents/cfd_code_6_sept/scaler.pkl")

    # Get user inputs
    user_data = {
        'Scenario': input("Enter the scenario: "),
        'Particle_size': input("Enter Particle Size: "),
        'Position_of_Freestanding_lights': input("Enter Position of Freestanding Lights: "),
        'Position_of_outlet': input("Enter Position of Outlet: "),
        'Temp': int(input("Enter Temperature: "))
       }
    
    input_data_scaled = preprocess_data(user_data)
        
    # Make a prediction    
    predictions = predict_ach_sum_of_ppm(model, input_data_scaled, scaler)    # Getting the predicted class and its confidence    
    print('predictions = ',predictions)
    print('predictions = ',predictions[0][0])
    predicted_class = np.argmax(predictions)  
    confidence_score = predictions[0][predicted_class]    
    ach_original = np.expm1(predictions[0][0])   
    ach_prediction = np.clip(ach_original.astype(int), 20, 26)   
    print('ach_prediction = ',ach_prediction)
    print('confidence_score = ',confidence_score * 10)    
    return ach_prediction.item(), confidence_score.item()
    
              
       
if __name__ == "__main__":
    main()
