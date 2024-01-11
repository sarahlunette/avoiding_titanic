import streamlit as st
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px 
import pybase64 as base64
import netCDF4
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

pages = ['Statistics on Icebreaker', 'Prediction of icebergs', 'Is Leonardo DiCaprio really dead ?']

st.sidebar.title('Navigation')
page = st.sidebar.radio("What\'s on your mind ?", pages)

#Import dataset and preprocess
df = pd.read_csv('AAD_Iceberg_database_1979-1984_version_2022-06-28.csv')
number_of_icebergs = df['Total'].sum()

df = df[['Obs_Date_NZ', 'Obs_Lat', 'Obs_Lon', 'Total', 'size1','size2', 'size3', 'size4', 'size5']]
df['Obs_Date_NZ'] = df['Obs_Date_NZ'].apply(str)
df['Obs_Date_NZ'][0]
df['Obs_Date_NZ']  = df['Obs_Date_NZ'].replace(r'\\', '-', regex=True)
df['Obs_Date_NZ'] = pd.to_datetime(df['Obs_Date_NZ'], dayfirst=True)

df['Total_observed'] = df['size1'] + df['size2']  + df['size3'] + df['size4'] + df['size5']
df['T'] = df['Total']-df['Total_observed']

import datetime as dt
df = df[~((df['size1'] == 0) & (df['size2'] == 0)& (df['size3'] == 0)& (df['size4'] == 0)& (df['size5'] == 0))]
df = df[df['T'].isin([0,1])]
df['size1'] = df['size1'].apply(lambda x: 0 if x == 0 else 1)
df['size2'] = df['size2'].apply(lambda x: 0 if x == 0 else 2)
df['size3'] = df['size3'].apply(lambda x: 0 if x == 0 else 3)
df['size4'] = df['size4'].apply(lambda x: 0 if x == 0 else 4)
df['size5'] = df['size5'].apply(lambda x: 0 if x == 0 else 5)
df['size'] = df[['size1', 'size2', 'size3', 'size4', 'size5']].max(axis = 1).values
df['size1'] = df['size1'].apply(lambda x: 0 if x == 0 else 1)
df['size2'] = df['size2'].apply(lambda x: 0 if x == 0 else 1)
df['size3'] = df['size3'].apply(lambda x: 0 if x == 0 else 1)
df['size4'] = df['size4'].apply(lambda x: 0 if x == 0 else 1)
df['size5'] = df['size5'].apply(lambda x: 0 if x == 0 else 1)
df.drop('Total_observed', axis = 1, inplace = True)
df['year'] = df['Obs_Date_NZ'].dt.year
df['month'] = df['Obs_Date_NZ'].dt.month
df = df.drop('Obs_Date_NZ', axis = 1)
df.drop('T', axis = 1, inplace = True)

X = df.drop('Total', axis = 1)
y = df['Total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

models_ = ['Linear Regression', 'Random Forest', 'Gradient Boosting Regressor' ]
models_size = ['Gradient Boosting Regressor', 'Gradient Boosting Classifier']

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
score_lr = lr.score(X_test, y_test)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
score_rf = rf.score(X_test, y_test)

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
score_gbr = gbr.score(X_test, y_test)

scores_ = [score_lr, score_rf, score_gbr]
X_size = df.drop('size', axis = 1)
y_size = df['size']
X_train, X_test, y_train, y_test = train_test_split(X_size, y_size, test_size = 0.3, random_state = 42)

from sklearn.ensemble import GradientBoostingClassifier
gbc_size = GradientBoostingClassifier()
gbc_size.fit(X_train, y_train)
y_pred = gbc_size.predict(X_test)
gbc_size.score(X_test, y_test)

from sklearn.ensemble import GradientBoostingRegressor
gbr_size = GradientBoostingRegressor()
gbr_size.fit(X_train, y_train)
y_pred = gbr_size.predict(X_test)
gbr_size.score(X_test, y_test)

Totals = df.groupby('year').count()['Total']
fig, ax = plt.subplots()
ax.plot(Totals.index, Totals.values)


if page == pages[0]:
    st.title("Avoiding Titanic")
    st.header('First Option')
    st.image('RescueHelicopter.jpg')
    st.header('Second Option')
    st.write('Icebreaker')

uploaded_files = st.file_uploader(
    "Choose a PDF file",
    accept_multiple_files=True,
    type=['pdf']
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()

    st.write("filename:", uploaded_file.name)
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)

if page == pages[1]:
    st.title('Copernicus Data')
    st.header('Statistics on the density of icebergs in the ocean')
    st.write('Number of icebergs in the dataset ' + str(number_of_icebergs))
    st.write("Check out this [link](https://data.marine.copernicus.eu/product/SEAICE_ARC_SEAICE_L4_NRT_OBSERVATIONS_011_007/description?view=-&product_id=-&option=-)")

    option = st.selectbox('Models', tuple(models_))
    st.write('You selected:', option)
    if option == models_[0]:
        st.write(str(scores_[0]))
    elif option == models_[1]:
        st.write(str(scores_[1]))
    elif option == models_[2]:
        st.write(str(scores_[2]))

#Then change to size models

if page == pages[2]:
    st.image('Leonardo.jpg')

