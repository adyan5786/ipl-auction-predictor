# Required packages: streamlit, pandas, joblib, numpy.

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="IPL Auction Price Predictor", page_icon="🏏")


@st.cache_resource
def load_artifacts():
	"""Load the trained model and preprocessing artifacts once for the app."""
	model = joblib.load("model/model.pkl")
	label_encoders = joblib.load("model/label_encoders.pkl")
	feature_columns = joblib.load("model/feature_columns.pkl")
	scaler = joblib.load("model/scaler.pkl")
	return model, label_encoders, feature_columns, scaler


model, label_encoders, feature_columns, scaler = load_artifacts()

st.title("IPL Auction Price Predictor")
st.write("Estimate an IPL player's auction price from historical performance data.")

st.sidebar.header("Player Details")
role = st.sidebar.selectbox("Role", label_encoders["Role"].classes_)
team = st.sidebar.selectbox("Team", label_encoders["Team"].classes_)
origin = st.sidebar.selectbox(
	"Player Origin", label_encoders["Player Origin"].classes_
)
# TODO: Confirm this range against the actual Year values in training data.
auction_year = st.sidebar.slider("Auction Year", min_value=2013, max_value=2022, value=2022)

# TODO: Fine-tune these ranges after checking min/max values in ipl_merged.csv.
st.sidebar.subheader("Batting Stats")
matches = st.sidebar.number_input("Matches", min_value=0, max_value=250, value=20)
runs = st.sidebar.number_input("Runs", min_value=0, max_value=6000, value=200)
boundaries = st.sidebar.number_input(
	"Boundaries", min_value=0, max_value=700, value=20
)
balls_faced = st.sidebar.number_input(
	"Balls Faced", min_value=0, max_value=4000, value=150
)
batting_avg = st.sidebar.number_input(
	"Batting Average", min_value=0.0, max_value=60.0, value=20.0
)
batting_strike_rate = st.sidebar.number_input(
	"Batting Strike Rate", min_value=0.0, max_value=200.0, value=120.0
)
boundaries_percent = st.sidebar.number_input(
	"Boundaries Percent", min_value=0.0, max_value=100.0, value=15.0
)

st.sidebar.subheader("Bowling & Fielding Stats")
wickets = st.sidebar.number_input("Wickets", min_value=0, max_value=200, value=5)
balls_bowled = st.sidebar.number_input(
	"Balls Bowled", min_value=0, max_value=5000, value=100
)
runs_conceded = st.sidebar.number_input(
	"Runs Conceded", min_value=0, max_value=5000, value=150
)
bowling_avg = st.sidebar.number_input(
	"Bowling Average", min_value=0.0, max_value=100.0, value=30.0
)
bowling_economy = st.sidebar.number_input(
	"Bowling Economy", min_value=0.0, max_value=15.0, value=8.0
)
bowling_strike_rate = st.sidebar.number_input(
	"Bowling Strike Rate", min_value=0.0, max_value=100.0, value=20.0
)
catches = st.sidebar.number_input("Catches", min_value=0, max_value=150, value=5)
stumpings = st.sidebar.number_input(
	"Stumpings", min_value=0, max_value=50, value=0
)

if st.sidebar.button("Predict Auction Price"):
	# Encode the selected categorical values using the saved training encoders.
	encoded_categories = {
		"Role": label_encoders["Role"].transform([role])[0],
		"Team": label_encoders["Team"].transform([team])[0],
		"Player Origin": label_encoders["Player Origin"].transform([origin])[0],
	}

	input_data = {column: 0 for column in feature_columns}
	input_data.update(encoded_categories)
	input_data.update(
		{
			"matches": matches,
			"runs": runs,
			"boundaries": boundaries,
			"balls_faced": balls_faced,
			"batting_avg": batting_avg,
			"batting_strike_rate": batting_strike_rate,
			"boundaries_percent": boundaries_percent,
			"wickets": wickets,
			"balls_bowled": balls_bowled,
			"runs_conceded": runs_conceded,
			"bowling_avg": bowling_avg,
			"bowling_economy": bowling_economy,
			"bowling_strike_rate": bowling_strike_rate,
			"catches": catches,
			"stumpings": stumpings,
		}
	)

	input_data["Year"] = auction_year

	input_df = pd.DataFrame([input_data]).reindex(columns=feature_columns)
	numeric_feature_columns = list(scaler.feature_names_in_)
	input_df[numeric_feature_columns] = scaler.transform(
		input_df[numeric_feature_columns]
	)
	input_df = input_df.reindex(columns=feature_columns)
	predicted_amount = float(np.asarray(model.predict(input_df))[0])
	predicted_crores = predicted_amount / 10_000_000
	st.success(f"Predicted Price: ₹ {predicted_crores:.2f} Crore")

st.info(
	"This is an academic project. Predictions are estimates based on historical "
	"data and are not guarantees of real auction prices."
)
