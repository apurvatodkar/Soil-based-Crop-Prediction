import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("soil_crop_data.csv")

# Inputs and outputs
X = data[['moisture', 'ph', 'temperature', 'humidity']]
soil = data['soil_type']
crop = data['crop']

# Encode labels
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

soil_encoded = soil_encoder.fit_transform(soil)
crop_encoded = crop_encoder.fit_transform(crop)

# Train-test split
X_train, X_test, soil_y_train, soil_y_test = train_test_split(
    X, soil_encoded, test_size=0.2, random_state=42
)

_, _, crop_y_train, crop_y_test = train_test_split(
    X, crop_encoded, test_size=0.2, random_state=42
)

# Train models
soil_model = RandomForestClassifier(n_estimators=100)
crop_model = RandomForestClassifier(n_estimators=100)

soil_model.fit(X_train, soil_y_train)
crop_model.fit(X_train, crop_y_train)

# Accuracy
soil_accuracy = accuracy_score(soil_y_test, soil_model.predict(X_test))
crop_accuracy = accuracy_score(crop_y_test, crop_model.predict(X_test))

def predict_soil_and_crop(moisture, ph, temperature, humidity):
    input_data = [[moisture, ph, temperature, humidity]]

    soil_pred = soil_model.predict(input_data)
    crop_pred = crop_model.predict(input_data)

    soil_result = soil_encoder.inverse_transform(soil_pred)[0]
    crop_result = crop_encoder.inverse_transform(crop_pred)[0]

    return soil_result, crop_result

def get_accuracies():
    return soil_accuracy, crop_accuracy

def get_dataset():
    return data
