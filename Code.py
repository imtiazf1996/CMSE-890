# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None

# Function to load and preprocess the dataset
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    # Remove rows where 'price' or 'mileage' is zero or missing
    df = df[(df['pricesold'] > 0) & (df['Mileage'] > 0) & (df['Mileage'] <= 400000) & (df['Year'] <= 2024) & (df['Year'] > 1980) & (df['pricesold'] < 100000)]
    # Convert 'year' to numeric and adjust 'mileage' and 'price'
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[~df['Model'].astype(str).str.contains('\$')]
    # Drop any remaining rows with missing values
    df.dropna(subset=['pricesold', 'Year', 'Mileage'], inplace=True)
    return df

# Load your dataset
file_path = 'used_car_sales.csv'  # Update with your file path
df = load_data(file_path)

# Streamlit user interface for EDA
st.title('Used Car Price Prediction - EDA')
plot_choice = st.selectbox(
    'Choose a plot to view:',
    ['Price Distribution', 'Mileage Distribution', 'Price vs. Year', 'Price vs. Mileage']
)

if plot_choice == 'Price Distribution':
    plt.figure(figsize=(8, 4))
    sns.histplot(df['pricesold'], bins=30, kde=True)
    plt.title('Distribution of Used Car Prices (USD)')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif plot_choice == 'Mileage Distribution':
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Mileage'], bins=30, kde=True)
    plt.title('Distribution of Car Mileage (miles)')
    plt.xlabel('Mileage (miles)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif plot_choice == 'Price vs. Year':
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Year', y='pricesold', data=df, alpha=0.6)
    plt.title('Price vs. Year')
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    st.pyplot(plt)

elif plot_choice == 'Price vs. Mileage':
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Mileage', y='pricesold', data=df, alpha=0.6)
    plt.title('Price vs. Mileage')
    plt.xlabel('Mileage (miles)')
    plt.ylabel('Price (USD)')
    st.pyplot(plt)


# Data preprocessing
def preprocess_data(data):
    features = ['Year', 'Mileage', 'Make', 'Model']  # Modify as per your dataset
    target = 'pricesold'

    X = data[features]
    y = data[target]

    numerical_features = ['Year', 'Mileage']
    categorical_features = ['Make', 'Model']  # Modify as per your dataset

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

    return preprocessor, X, y, features, numerical_features,categorical_features

preprocessor, X, y, features, numerical_features,categorical_features = preprocess_data(df)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection Interface
st.header('Model Selection and Training')
model_choice = st.selectbox(
    'Select a machine learning model:',
    ['Neural Network (MLPRegressor)', 'Support Vector Machine (SVR)', 'Linear Regression', 'Random Forest']
)

# Display model-specific hyperparameter options
if model_choice == 'Neural Network (MLPRegressor)':
    layer_sizes = st.text_input('Enter hidden layer sizes (e.g., 100,50)', '100')
    max_iter = st.slider('Max iterations', min_value=100, max_value=1000, value=666, step=50)
    alpha = st.text_input('Enter alpha (e.g., 0.001)', '0.001006064345328208')
    learning_rate_init = st.text_input('Enter learning rate init (e.g., 0.003)', '0.0029967378215835973')


elif model_choice == 'Random Forest':
    n_estimators = st.slider('Number of trees', min_value=10, max_value=300, value=100, step=10)
    max_depth = st.slider('Maximum depth of the trees', min_value=1, max_value=50, value=5, step=1)

elif model_choice == 'Support Vector Machine (SVR)':
    C = st.slider('C (Regularization parameter)', min_value=0.01, max_value=100.0, value=1.0, step=0.01)
    kernel = st.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

# Train and evaluate the selected model
if st.button('Train and Evaluate Model'):
    # Initialize the model based on user selection
    if model_choice == 'Neural Network (MLPRegressor)':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MLPRegressor(hidden_layer_sizes=tuple(map(int, layer_sizes.split(','))), 
                                       max_iter=max_iter, random_state=42))
        ])
    elif model_choice == 'Random Forest':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
        ])

    elif model_choice == 'Support Vector Machine (SVR)':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR(C=C, kernel=kernel))
        ])
    elif model_choice == 'Linear Regression':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

    # Train the model
    model.fit(X_train, y_train)
    
    # car predictions and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)**0.5
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = X_test.shape[0]  # number of observations
    p = X_test.shape[1]  # number of predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    med_ae = median_absolute_error(y_test, y_pred)

    # Display evaluation metrics and model information
    st.write(f'Model selected: {model_choice}')
    st.write(f'MSE (Mean Squared Error): {mse:.3f}')
    st.write(f'RMSE (Root Mean Squared Error): {rmse:.3f}')
    st.write(f'MAE (Mean Absolute Error): {mae:.3f}')
    st.write(f'R² score: {r2:.3f}')
    st.write(f'Adjusted R² score: {adj_r2:.3f}')
    st.write(f'Median AE (Median Absolute Error): {med_ae:.3f}')
        
    # Save the trained model in session state
    st.session_state.model_trained = True
    st.session_state.model = model
    
    # Plotting actual vs predicted prices for evaluation
    st.header("Model Evaluation Visualizations")

    st.subheader("Actual vs. Predicted Prices")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    ax.set_xlabel('Actual Prices')
    ax.set_ylabel('Predicted Prices')
    st.pyplot(fig)
    
    # Visualization: Residual Plot
    st.subheader("Residuals vs. Actual Prices")
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=residuals, ax=ax)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Actual Prices')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)
    
    # Visualization: Distribution of Residuals
    st.subheader("Distribution of Residuals")
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    if model_choice == 'Random Forest':
    # Feature importance plot
        importances = model.named_steps['regressor'].feature_importances_
    # Correctly get feature names for categorical variables
        feature_names = numerical_features + \
            list(preprocessor.named_transformers_['cat'].onehot.get_feature_names(categorical_features))
        forest_importances = pd.Series(importances, index=feature_names)
        fig, ax = plt.subplots()
        forest_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances based on mean decrease in impurity")
        ax.set_ylabel("Importance")
        st.pyplot(fig)

# Adjust the 'Predict New Data' section to meet your requirements
st.header("Predict New Car Prices")

# Initialize an empty dictionary to hold user inputs
input_data = {}

# Year selection limited between 1980 and 2024
input_data['Year'] = st.number_input('Enter Year', min_value=1980, max_value=2024, value=2020, step=1)

# Make selection
unique_makes = df['Make'].unique()
selected_make = st.selectbox('Select Make', options=unique_makes)

# Model selection based on selected Make
if selected_make:
    unique_models = df[df['Make'] == selected_make]['Model'].unique()
    selected_model = st.selectbox('Select Model', options=unique_models)
    input_data['Make'] = selected_make  # Store the selected make
    input_data['Model'] = selected_model  # Store the selected model

# Mileage input
input_data['Mileage'] = st.number_input('Enter Mileage', min_value=0, max_value=400000, value=50000, step=1000)

# Specify the expected columns in the order they are expected
expected_columns = ['Year', 'Mileage', 'Make', 'Model']

# Create a DataFrame from input_data
input_df = pd.DataFrame([input_data], columns=expected_columns)


# Predict Price button
if st.button('Predict Price'):
    if st.session_state.get('model_trained', False):
        model = st.session_state.model  # Ensure this model includes the preprocessor
        # Predict using the model
        prediction = model.predict(input_df)  # input_df is now correctly structured for prediction
        st.write(f"Predicted Price: ${prediction[0]:,.2f}")
    else:
        st.error("Please train the model before predicting.")
