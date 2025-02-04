import pandas as pd
import streamlit as st
import requests
from datetime import timedelta


from src.model import AnalysisResult


def fetch_ds_data(token):
    url = "https://www.dreamingspanish.com/.netlify/functions/dayWatchedTime"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def load_data(token):
    """Fetch and process data from API"""
    if not token or not token.strip():
        return None

    api_data = fetch_ds_data(token)
    if not api_data or not isinstance(api_data, list) or len(api_data) == 0:
        return None

    # Convert API data to DataFrame
    df = pd.DataFrame(api_data)
    df["date"] = pd.to_datetime(df["date"])

    # Create a complete date range
    df = df.set_index("date").asfreq("D").reset_index()
    df = df.rename(columns={"index": "date"})

    # Fill missing values with explicit types
    df = df.astype({"timeSeconds": "float64", "goalReached": "boolean"}) \
        .fillna({"timeSeconds": 0.0, "goalReached": False})

    # Sort by date
    df = df.sort_values("date")

    # Add goal tracking metrics
    total_days = len(df)
    goals_reached = df["goalReached"].sum()

    # Calculate current goal streak
    df["goal_streak_group"] = (~df["goalReached"]).cumsum()
    df["current_goal_streak"] = df.groupby("goal_streak_group")[
        "goalReached"].cumsum()
    current_goal_streak = (
        df["current_goal_streak"].iloc[-1] if df["goalReached"].iloc[-1] else 0
    )

    # Calculate longest goal streak
    goal_streak_lengths = df[df["goalReached"]
                             ].groupby("goal_streak_group").size()
    longest_goal_streak = (
        goal_streak_lengths.max() if not goal_streak_lengths.empty else 0
    )

    return AnalysisResult(df=df, goals_reached=goals_reached, total_days=total_days, current_goal_streak=current_goal_streak, longest_goal_streak=longest_goal_streak)


def generate_future_predictions(df, avg_seconds_per_day, days_to_predict=800):
    if len(df) == 0:
        return pd.DataFrame()

    if avg_seconds_per_day <= 0:
        avg_seconds_per_day = 1  # Prevent division by zero

    last_date = df["date"].iloc[-1]
    # Start future dates from the day after the last historical date
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=days_to_predict, freq="D")

    future_seconds = pd.Series([avg_seconds_per_day] * len(future_dates))
    future_df = pd.DataFrame({"date": future_dates, "seconds": future_seconds})

    # Start cumulative seconds from the last cumulative seconds of the historical data
    last_cumulative_seconds = df["cumulative_seconds"].iloc[-1]
    future_df["cumulative_seconds"] = future_seconds.cumsum() + \
        last_cumulative_seconds
    future_df["cumulative_minutes"] = future_df["cumulative_seconds"] / 60
    future_df["cumulative_hours"] = future_df["cumulative_minutes"] / 60

    # Create a single row DataFrame for the last historical point
    last_point = pd.DataFrame({
        "date": [last_date],
        "seconds": [df["seconds"].iloc[-1]],
        "cumulative_seconds": [last_cumulative_seconds],
        "cumulative_minutes": [last_cumulative_seconds / 60],
        "cumulative_hours": [last_cumulative_seconds / 3600]
    })

    # Combine last historical point with future predictions
    combined_df = pd.concat([last_point, future_df], ignore_index=True)

    return combined_df
