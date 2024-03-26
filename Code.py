# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None

# Function to load and preprocess the dataset
@st.cache
def load_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    # Remove rows where 'price' or 'mileage' is zero or missing
    df = df[(df['price'] > 0) & (df['mileage'] > 0)]
    # Convert 'year' to numeric and adjust 'mileage' and 'price'
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['mileage'] = df['mileage'] / 2  # Convert mileage to miles
    df['price'] = df['price'] * 500  # Convert price to USD
    # Drop any remaining rows with missing values
    df.dropna(subset=['price', 'year', 'mileage'], inplace=True)
    return df

# Load your dataset
file_path = 'car_sales_us.csv'  # Update with your file path
df = load_data(file_path)

# Streamlit user interface for EDA
st.title('Used Car Price Prediction - EDA')
plot_choice = st.selectbox(
    'Choose a plot to view:',
    ['Price Distribution', 'Mileage Distribution', 'Price vs. Year', 'Price vs. Mileage']
)

if plot_choice == 'Price Distribution':
    plt.figure(figsize=(8, 4))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Distribution of Used Car Prices (USD)')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif plot_choice == 'Mileage Distribution':
    plt.figure(figsize=(8, 4))
    sns.histplot(df['mileage'], bins=30, kde=True)
    plt.title('Distribution of Car Mileage (miles)')
    plt.xlabel('Mileage (miles)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif plot_choice == 'Price vs. Year':
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='year', y='price', data=df, alpha=0.6)
    plt.title('Price vs. Year')
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    st.pyplot(plt)

elif plot_choice == 'Price vs. Mileage':
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='mileage', y='price', data=df, alpha=0.6)
    plt.title('Price vs. Mileage')
    plt.xlabel('Mileage (miles)')
    plt.ylabel('Price (USD)')
    st.pyplot(plt)


# Data preprocessing
def preprocess_data(data):
    features = ['year', 'mileage', 'car', 'model']  # Modify as per your dataset
    target = 'price'

    X = data[features]
    y = data[target]

    numerical_features = ['year', 'mileage']
    categorical_features = ['car', 'model']  # Modify as per your dataset

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

# Model Selection Interface
st.header('Model Selection and Training')
model_choice = st.selectbox(
    'Select a machine learning model:',
    ['Neural Network (MLPRegressor)', 'Support Vector Machine (SVR)', 'Linear Regression']
)

# Display model-specific hyperparameter options
if model_choice == 'Neural Network (MLPRegressor)':
    layer_sizes = st.text_input('Enter hidden layer sizes (e.g., 100,50)', '100,50')
    max_iter = st.slider('Max iterations', min_value=100, max_value=1000, value=500, step=50)
elif model_choice == 'Support Vector Machine (SVR)':
    C = st.slider('C (Regularization parameter)', min_value=0.01, max_value=10.0, value=1.0, step=0.01)
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
    r2 = r2_score(y_test, y_pred)
    
    # Display evaluation metrics and model information
    st.write(f'Model selected: {model_choice}')
    st.write(f'R² score: {r2:.3f}')
    
    # Save the trained model in session state
    st.session_state.model_trained = True
    st.session_state.model = model
    
    # Plotting actual vs predicted prices for evaluation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual Prices vs Predicted Prices')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line for reference
    st.pyplot(plt)

