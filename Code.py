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

# Include additional sections as needed (model training, evaluation, user input prediction, etc.)
# Refer to the previous segments for adding those functionalities if required.
