# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Function to load and preprocess the dataset
@st.cache  # This function will cache the data so it only loads once
def load_data(filepath):
    # Specify encoding to avoid 'utf-8' codec errors
    df = pd.read_csv(filepath, encoding='ISO-8859-1')  # Adjust encoding if necessary
    df = df[df['price'] > 0]
    df.dropna(subset=['price', 'year', 'mileage'], inplace=True)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df.dropna(subset=['year'], inplace=True)
    return df

# Load your dataset
file_path = 'car_sales_us.csv'  # Replace with your file path
df = load_data(file_path)

# Streamlit user interface
st.title('Used Car Price Prediction')
model_choice = st.selectbox('Choose a machine learning model:', ['Neural Network', 'Support Vector Machine'])

# Data preprocessing
def preprocess_data(data):
    features = ['make', 'model', 'year', 'mileage', 'engV', 'engType', 'body', 'drive']
    target = 'price'

    X = data[features]
    y = data[target]

    numerical_features = ['year', 'mileage', 'engV']
    categorical_features = ['make', 'model', 'engType', 'body', 'drive']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor, X, y

preprocessor, X, y = preprocess_data(df)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the model based on user choice
if st.button('Train and Evaluate Model'):
    if model_choice == 'Neural Network':
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', MLPRegressor(hidden_layer_sizes=(50, 30), random_state=42, max_iter=500))])
    elif model_choice == 'Support Vector Machine':
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', SVR(C=1.0, epsilon=0.2))])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f'Model selected: {model_choice}')
    st.write(f'R² score: {r2:.2f}')
