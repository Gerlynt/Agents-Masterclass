import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from groq import Groq
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="📈 AI Forecasting Agent", page_icon="🤖", layout="wide")
st.title("📈 Revenue Forecasting using Prophet + AI Commentary")

st.markdown("Upload an Excel file with **Date** and **Revenue** columns:")

uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Clean and rename for Prophet
        df = df.rename(columns={"Date": "ds", "Revenue": "y"})
        df["ds"] = pd.to_datetime(df["ds"])

        st.subheader("📊 Raw Data Preview")
        st.dataframe(df)

        # Prophet model
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        st.subheader("🔮 Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("🧠 AI-Generated Forecast Commentary")

        # Send forecast data to AI model
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_dict(orient="records")
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
        You are a senior financial analyst. Analyze the revenue forecast based on the following data:
        {forecast_data}

        Provide:
        - Key trends and observations in the forecast.
        - Any risks or opportunities in the revenue outlook.
        - Recommendations for business strategy.
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a SaaS financial forecasting expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        ai_summary = response.choices[0].message.content
        st.write(ai_summary)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Awaiting file upload to begin forecasting...")
