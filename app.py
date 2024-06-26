import streamlit as st
import pandas as pd
import pickle

scalers = {
    'StandardScaler': pickle.load(open('StandardScaler.pkl', 'rb')),
    'MinMaxScaler': pickle.load(open('MinMaxScaler.pkl', 'rb')),
    'RobustScaler': pickle.load(open('RobustScaler.pkl', 'rb'))
}

models = {
    'LinearRegression': pickle.load(open('LinearRegression_StandardScaler.pkl', 'rb')),
    'Ridge': pickle.load(open('Ridge_StandardScaler.pkl', 'rb')),
    'Lasso': pickle.load(open('Lasso_StandardScaler.pkl', 'rb')),
    'DecisionTreeRegressor': pickle.load(open('DecisionTreeRegressor_StandardScaler.pkl', 'rb')),
    'RandomForestRegressor': pickle.load(open('RandomForestRegressor_StandardScaler.pkl', 'rb')),
    # 'GradientBoostingRegressor': pickle.load(open('GradientBoostingRegressor_StandardScaler.pkl', 'rb')),
    'SVR': pickle.load(open('SVR_StandardScaler.pkl', 'rb'))
}

st.set_page_config(page_title='Boston Housing Price Predictor', layout='wide')

st.title('Boston Housing Price Predictor')
st.write("Choose a scaler and a model to see the results. You can also input your own data for prediction.")

st.sidebar.header('Model and Scaler Selection')
scaler_choice = st.sidebar.selectbox('Scaler', list(scalers.keys()))
model_choice = st.sidebar.selectbox('Model', list(models.keys()))
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
st.header('Prediction Input and Results')
input_df = user_input_features()
if st.button('Predict'):
    scaler = scalers[scaler_choice]
    input_scaled = scaler.transform(input_df)
    model = models[model_choice]
    prediction = model.predict(input_scaled)
    st.subheader('Prediction Results')
    st.write(f'Model used: **{model_choice}**')
    st.write(f'Scaler used: **{scaler_choice}**')
    st.write(f'MEDV (Median value of owner-occupied homes in k dollars ): ** {prediction[0]:.2f} k dollars')
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
