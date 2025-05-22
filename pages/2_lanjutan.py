import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Prediction", page_icon="📈")
st.header("Predictions")

# Load Dataset
df = pd.read_csv("model/iris.csv")
dataset = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/.../iris.csv") 

# Tampilkan Dataframe
st.subheader("📊 Iris Dataset")
st.dataframe(df)
