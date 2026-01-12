import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Inputs and outputs
X = data[['N','P','K', 'temperature', 'humidity','ph','rainfall']]
crop = data['label']

# Encode labels
crop_encoder = LabelEncoder()
crop_encoded = crop_encoder.fit_transform(crop)

# Train-test split

X_train,X_test, crop_y_train, crop_y_test = train_test_split(
    X, crop_encoded, test_size=0.2, random_state=42
)

# Train models

crop_model = RandomForestClassifier(n_estimators=100)


crop_model.fit( X_train,crop_y_train)

# Accuracy
crop_accuracy = accuracy_score(crop_y_test, crop_model.predict(X_test))

def predict_soil_and_crop(N,P,K, temperature, humidity,ph,rainfall):
    input_data = [[N,P,K, temperature, humidity,ph,rainfall]]

  
    crop_pred = crop_model.predict(input_data)


    crop_result = crop_encoder.inverse_transform(crop_pred)[0]

    return  crop_result

def get_accuracies():
    return  crop_accuracy

def get_dataset():
    return data
