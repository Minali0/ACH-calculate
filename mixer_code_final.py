import pandas as pd
import re
import os
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Function to process a single 'prof' file and return a DataFrame.
def process_prof_file(input_file_path):
    # Create an empty list to hold the data from the 'prof' file.
    data = []

    # Open the 'prof' file for reading.
    with open(input_file_path, 'r') as prof_file:
        # Assuming the 'prof' files contain tab-separated values (TSV), change the delimiter accordingly if needed.
        delimiter = '\t'
        # Read each line and split it using the delimiter.
        for line in prof_file:
            # Assuming the data is organized in rows and each row contains multiple fields.
            row = line.strip().split(delimiter)
            data.append(row)

        # Remove the header row from the data (if it exists).
        data = data[1:]
        # Use regex to remove parentheses (,) from the data.
        for i in range(len(data)):
            for j in range(len(data[i])):
                # Remove parentheses from each field.
                data[i][j] = re.sub(r'[(),]', '', data[i][j])

            # Remove blank rows from the data.
        data = [row for row in data if any(field.strip() for field in row)]

    # Separate the data into columns
    x_data = []
    y_data = []
    z_data = []
    ppm_data = []

    current_column = []

    for item in data:
        value = item[0]  # Extract the string value from the inner list
        if value == 'x':
            current_column = x_data
        elif value == 'y':
            current_column = y_data
        elif value == 'z':
            current_column = z_data
        elif value == 'ppm':
            current_column = ppm_data
        else:
            current_column.append(float(value))  # Convert the string value to a float and append

    # Create a dictionary containing the data columns
    data_dict = {
        'x': x_data,
        'y': y_data,
        'z': z_data,
        'ppm': ppm_data,
    }

     # Create a DataFrame using pandas
    df = pd.DataFrame(data_dict)
    # Calculate the sum of 'ppm' values and add it as a new column
    df['Sum_of_ppm'] = round(df['ppm'].sum(),2)

    return df

def preprocess_data(df):
    # Define the features (X) and the target variable (y)
    X = df[['Scenario', 'Particle_size', 'Position_of_Freestanding_lights', 'Position_of_outlet', 'Temp']]
    y = df[['ACH','Sum_of_ppm']]

    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Reorder columns to match the desired order
    desired_column_order = ['Scenario', 'Particle_size', 'Position_of_Freestanding_lights', 'Position_of_outlet', 'Temp']
    X = X[desired_column_order]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Normalize the target variables (log transformation)
    y_train_normalized = np.log1p(y_train)  # Adding 1 before the log to avoid log(0)
    y_test_normalized = np.log1p(y_test)
    print("y_train_normalised :",y_train_normalized)
    print("y_test_normalised :",y_test_normalized)

    scaler_save_path = "/home/seaflux/Documents/cfd_code_6_sept/scaler.pkl"
    joblib.dump(scaler, scaler_save_path)

    return X_train, X_test, y_train_normalized, y_test_normalized, scaler
  
def build_model(input_shape):
    # Build the ANN model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train_normalized, epochs, batch_size, verbose=1):

    # Train the model on the training data
    model.fit(X_train, y_train_normalized, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Save the trained model
    model_save_path = "/home/seaflux/Documents/cfd_code_6_sept/model.keras"
    model.save(model_save_path)
    print("Model saved to:", model_save_path)

def predict_ach_sum_of_ppm(model, input_data, scaler):
    # Preprocess input data and make a prediction
    input_data_scaled = scaler.transform(input_data)
    predictions = model.predict(input_data_scaled).flatten() 
    return predictions[0], predictions[1] 

def main():

    # Replace 'path_to_your_zip_file.zip' with the actual path to your zip file
    zip_file_path = '/home/seaflux/Documents/cfd_code_6_sept/dataset.zip'

    # Replace 'path_to_extract' with the directory where you want to extract the contents
    extract_folder_name = os.path.splitext(os.path.basename(zip_file_path))[0]

    # Set the extraction and input folder paths based on the extracted folder name
    extract_folder_path = os.path.join('/home/seaflux/Documents/cfd_code_6_sept/', extract_folder_name)
    input_folder_path = os.path.join(extract_folder_path, extract_folder_name)

    # Create the directory if it doesn't exist
    os.makedirs(extract_folder_path, exist_ok=True)

    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder_path)

    # Get a list of all 'prof' files in the input folder
    prof_files = [f for f in os.listdir(input_folder_path) if f.endswith('.prof')]

    # Lists to store the extracted information as tuples
    data_rows = []

    # Loop through each 'prof' file
    for prof_file in prof_files:
        input_file_path = os.path.join(input_folder_path, prof_file)
        df = process_prof_file(input_file_path)

        # Extract information from the filename
        elements = prof_file.split('.')
        # Extract information from the elements
        Serial = int(elements[0])
        Particle_size = elements[1]
        Position_of_Freestanding_lights = tuple(elements[2:4])
        Position_of_outlet = elements[-5]
        ACH =  int(elements[-4])
        Temp = int(elements[-3])
        Scenario = int(elements[-2])

        # Map 'Position_of_outlet' values ('a', 'b', 'c') to numerical values (1, 2, 3)
        Position_of_outlet_numerical = {'a': 1, 'b': 2, 'c': 3}.get(Position_of_outlet, -1)
        
        
        Particle_size_numerical = {
            '07': 0.7,
            '05': 0.5,
            '04': 0.4,
            '01': 0.1,
            '0045': 0.045,
            '1': 1.0,
            '15': 1.5,
            '25': 2.5,
            '2': 2.0,
            '3': 3.0,
        }.get(Particle_size, -1)

        # Map 'Position_of_Freestanding_lights' values to numerical values
        Position_of_Freestanding_lights_numerical = {
            ('h'):1,
            ('t'):2,
            ('f'):3,
            ('h', 'f'): 4,
            ('h', 't'): 5,
            ('t', 'f'): 6,
            ('h','t','f'): 7
            
        }.get(Position_of_Freestanding_lights, -1)

        # Get the sum of 'ppm' from the 'Sum_of_ppm' column of the DataFrame
        Sum_of_ppm = df['Sum_of_ppm'].iloc[0]

        # Append the extracted information as a tuple
        data_rows.append((Serial,Scenario,Particle_size_numerical, Position_of_Freestanding_lights_numerical,
                        Position_of_outlet_numerical,Temp ,ACH,Sum_of_ppm))

    # Sort the list of tuples based on Serial and Scenario order
    data_rows.sort(key=lambda x: (x[0], x[1]))

    # Create a DataFrame using pandas
    df = pd.DataFrame(data_rows, columns=['Serial', 'Scenario', 'Particle_size', 'Position_of_Freestanding_lights',
                                        'Position_of_outlet', 'Temp', 'ACH', 'Sum_of_ppm'])

    # df.to_csv("/home/seaflux/Documents/latest_cfd_code/data.csv", index=False)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    input_shape = (X_train.shape[1],)
    model = build_model(input_shape)
    model.summary()
    train_model(model, X_train, y_train, epochs=250, batch_size=45, verbose=1)

    # Evaluate the model on the training set
    train_predictions = model.predict(X_train)
    train_mse = np.mean(np.square(train_predictions - y_train))
    train_rmse = np.sqrt(train_mse)
    print("Training MSE:", train_mse)
    print("Training RMSE:", train_rmse)

    # Evaluate the model on the testing set
    test_predictions = model.predict(X_test)
    test_mse = np.mean(np.square(test_predictions - y_test))
    test_rmse = np.sqrt(test_mse)
    print("Testing MSE:", test_mse)
    print("Testing RMSE:", test_rmse)



if __name__ == "__main__":
    main()   
