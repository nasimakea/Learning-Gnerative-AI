import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score

# Load trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("Machine Learning Model Deployment")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)
    LabelEncoder1=LabelEncoder()
    df['species']=LabelEncoder1.fit_transform(df['species'])
    Y_test=df['species']
    df.drop('species',axis=1,inplace=True)

    # Ensure the uploaded file has the same features
    if set(df.columns) != set(model.feature_names_in_):
        st.error("Uploaded file does not match trained model features.")
    else:
        # Make predictions
        predictions = model.predict(df)
        df["Predictions"] = predictions

        st.subheader("Predictions")
        st.dataframe(df)
        st.write("The accuracy is :",accuracy_score(predictions,Y_test))

        # Save Predictions
        df.to_csv("data.csv", index=False)
        st.download_button("Download Predictions", "data/predictions.csv")
