import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and scalers
scalers = {
    'StandardScaler': joblib.load('StandardScaler.pkl'),
    'MinMaxScaler': joblib.load('MinMaxScaler.pkl'),
    'RobustScaler': joblib.load('RobustScaler.pkl')
}

models = {
    'LinearRegression': joblib.load('LinearRegression_StandardScaler.pkl'),
    'Ridge': joblib.load('Ridge_StandardScaler.pkl'),
    'Lasso': joblib.load('Lasso_StandardScaler.pkl'),
    'DecisionTreeRegressor': joblib.load('DecisionTreeRegressor_StandardScaler.pkl'),
    'RandomForestRegressor': joblib.load('RandomForestRegressor_StandardScaler.pkl'),
    'GradientBoostingRegressor': joblib.load('GradientBoostingRegressor_StandardScaler.pkl'),
    'SVR': joblib.load('SVR_StandardScaler.pkl')
}

# Set page config
st.set_page_config(page_title='Boston Housing Price Predictor', layout='wide')

# Title
st.title('Boston Housing Price Predictor')
st.write("Choose a scaler and a model to see the results. You can also input your own data for prediction.")

# Sidebar for scaler and model selection
st.sidebar.header('Model and Scaler Selection')
scaler_choice = st.sidebar.selectbox('Scaler', list(scalers.keys()))
model_choice = st.sidebar.selectbox('Model', list(models.keys()))

# Sidebar for user input
st.sidebar.header('User Input Features')
def user_input_features():
    CRIM = st.sidebar.number_input('Per capita crime rate by town (CRIM)', value=0.00632)
    ZN = st.sidebar.number_input('Proportion of residential land zoned for lots over 25,000 sq.ft. (ZN)', value=18.0)
    INDUS = st.sidebar.number_input('Proportion of non-retail business acres per town (INDUS)', value=2.31)
    CHAS = st.sidebar.selectbox('Charles River dummy variable (CHAS)', [0, 1])
    NOX = st.sidebar.number_input('Nitric oxides concentration (NOX)', value=0.538)
    RM = st.sidebar.number_input('Average number of rooms per dwelling (RM)', value=6.575)
    AGE = st.sidebar.number_input('Proportion of owner-occupied units built prior to 1940 (AGE)', value=65.2)
    DIS = st.sidebar.number_input('Weighted distances to five Boston employment centres (DIS)', value=4.0900)
    RAD = st.sidebar.number_input('Index of accessibility to radial highways (RAD)', value=1)
    TAX = st.sidebar.number_input('Full-value property-tax rate per $10,000 (TAX)', value=296.0)
    PTRATIO = st.sidebar.number_input('Pupil-teacher ratio by town (PTRATIO)', value=15.3)
    B = st.sidebar.number_input('Proportion of blacks by town (B)', value=396.90)
    LSTAT = st.sidebar.number_input('Percentage of lower status of the population (LSTAT)', value=4.98)
    
    data = {
        'CRIM': CRIM,
        'ZN': ZN,
        'INDUS': INDUS,
        'CHAS': CHAS,
        'NOX': NOX,
        'RM': RM,
        'AGE': AGE,
        'DIS': DIS,
        'RAD': RAD,
        'TAX': TAX,
        'PTRATIO': PTRATIO,
        'B': B,
        'LSTAT': LSTAT
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Main page for displaying results
st.header('Prediction Input and Results')

# Get user inputs
input_df = user_input_features()

# Add a predict button
if st.button('Predict'):
    # Scale the input features
    scaler = scalers[scaler_choice]
    input_scaled = scaler.transform(input_df)

    # Load the chosen model and make predictions
    model = models[model_choice]
    prediction = model.predict(input_scaled)

    st.subheader('Prediction Results')
    st.write(f'Model used: **{model_choice}**')
    st.write(f'Scaler used: **{scaler_choice}**')
    st.write(f'MEDV (Median value of owner-occupied homes in dollars): **${prediction[0]:.2f}**')

# Layout and styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='sidebar .sidebar-content'>", unsafe_allow_html=True)
