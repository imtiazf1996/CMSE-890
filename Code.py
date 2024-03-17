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
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and preprocess the dataset
@st.cache
def load_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')  # Adjust encoding if necessary
    df = df[df['price'] > 0]
    df.dropna(subset=['price', 'year', 'mileage'], inplace=True)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df.dropna(subset=['year'], inplace=True)
    return df

# Load your dataset
file_path = 'car_sales_us.csv'  # Update with your file path
df = load_data(file_path)

# Streamlit user interface for Feature Exploration
st.title('Used Car Price Prediction')

if st.checkbox('Show Data Distribution'):
    feature_to_explore = st.selectbox('Which feature would you like to explore?', ['year', 'mileage', 'engV', 'price'])
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature_to_explore], kde=True)
    st.pyplot(plt)

# Data preprocessing
def preprocess_data(data):
    features = ['car', 'model', 'year', 'mileage', 'engV', 'engType', 'body', 'drive']
    target = 'price'

    X = data[features]
    y = data[target]

    numerical_features = ['year', 'mileage', 'engV']
    categorical_features = ['car', 'model', 'engType', 'body', 'drive']

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

# Allow user to choose machine learning model and set parameters
model_choice = st.selectbox('Choose a machine learning model:', ['Neural Network', 'Support Vector Machine'])

# Define models based on user choice
if model_choice == 'Neural Network':
    nn_model = MLPRegressor(random_state=42, max_iter=500)
elif model_choice == 'Support Vector Machine':
    svm_model = SVR()

# User Input Prediction with dynamic model selection based on make
if st.checkbox('Predict Your Car’s Price'):
    user_data = {}
    user_data['car'] = st.selectbox('Car Make', df['car'].unique())
    filtered_models = df[df['car'] == user_data['car']]['model'].unique()
    user_data['model'] = st.selectbox('Car Model', filtered_models)
    user_data['year'] = st.slider('Year', int(df['year'].min()), int(df['year'].max()), step=1)
    user_data['mileage'] = st.number_input('Mileage', min_value=0)
    user_data['engV'] = st.number_input('Engine Volume', min_value=0.0, step=0.1)
    user_data['engType'] = st.selectbox('Engine Type', df['engType'].unique())
    user_data['body'] = st.selectbox('Body Type', df['body'].unique())
    user_data['drive'] = st.selectbox('Drive Type', df['drive'].dropna().unique())
    
    user_features = pd.DataFrame([user_data])
    user_features_preprocessed = preprocessor.transform(user_features)
    
    if st.button('Predict Price'):
        if model_choice == 'Neural Network':
            model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', nn_model)])
        elif model_choice == 'Support Vector Machine':
            model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', svm_model)])
        
        model.fit(X_train, y_train)  # Train the model with training data
        predicted_price = model.predict(user_features_preprocessed)  # Predict price for user's car
        st.write(f"Estimated Price: ${predicted_price[0]:,.2f}")  # Display predicted price

# Train and evaluate the model based on user choice
if st.button('Train and Evaluate Selected Model'):
    if model_choice == 'Neural Network':
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', nn_model)])
    elif model_choice == 'Support Vector Machine':
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', svm_model)])
    
    model.fit(X_train, y_train)  # Train the model with training data
    y_pred = model.predict(X_test)  # Predict prices for test data
    r2 = r2_score(y_test, y_pred)  # Calculate R^2 score
    st.write(f'Model selected: {model_choice}')
    st.write(f'R² score: {r2:.2f}')  # Display R^2 score
