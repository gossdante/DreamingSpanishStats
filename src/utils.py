"""
Provides utility functions for managing and analyzing viewing progress data.
Includes methods for data processing, statistical insights, and forecasting future learning milestones
"""

from datetime import timedelta

import pandas as pd
import requests
import streamlit as st

from src.model import AnalysisResult


def fetch_ds_data(token: str) -> dict | None:
    """
    Fetches data from the Dreaming Spanish API using the provided bearer token.

    Args:
        token (str): The bearer token for API authentication.

    Returns:
        dict or None: A dictionary containing the fetched data if successful, otherwise None.
    """
    url = "https://www.dreamingspanish.com/.netlify/functions/dayWatchedTime"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def load_data(token: str) -> AnalysisResult | None:
    """
    Loads and processes data from the Dreaming Spanish API.

    This function fetches data using the provided bearer token, converts the data into a DataFrame,
    and processes it to include goal tracking metrics such as total days, goals reached, current goal streak,
    and longest goal streak.

    Args:
        token (str): The bearer token for API authentication.

    Returns:
        AnalysisResult or None: An AnalysisResult object containing the processed data if successful, otherwise None.
    """
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
    df = df.astype({"timeSeconds": "float64", "goalReached": "boolean"}).fillna(
        {"timeSeconds": 0.0, "goalReached": False}
    )

    # Sort by date
    df = df.sort_values("date")

    # Add goal tracking metrics
    total_days = len(df)
    goals_reached = df["goalReached"].sum()

    # Calculate current goal streak
    df["goal_streak_group"] = (~df["goalReached"]).cumsum()
    df["current_goal_streak"] = df.groupby("goal_streak_group")["goalReached"].cumsum()
    current_goal_streak = (
        df["current_goal_streak"].iloc[-1] if df["goalReached"].iloc[-1] else 0
    )

    # Calculate longest goal streak
    goal_streak_lengths = df[df["goalReached"]].groupby("goal_streak_group").size()
    longest_goal_streak = (
        goal_streak_lengths.max() if not goal_streak_lengths.empty else 0
    )

    return AnalysisResult(
        df=df,
        goals_reached=goals_reached,
        total_days=total_days,
        current_goal_streak=current_goal_streak,
        longest_goal_streak=longest_goal_streak,
    )


def generate_future_predictions(
    df: pd.DataFrame, avg_seconds_per_day: float, days_to_predict: int = 800
):
    """
    Generates future predictions based on historical data and a given average seconds per day.

    Args:
        df (pd.DataFrame): The existing DataFrame containing historical data.
        avg_seconds_per_day (float): The average seconds watched per day.
        days_to_predict (int, optional): The number of future days to predict. Defaults to 800.

    Returns:
        pd.DataFrame: A DataFrame containing future predictions with dates, seconds, cumulative seconds, cumulative minutes, and cumulative hours.
    """
    if len(df) == 0:
        return pd.DataFrame()

    if avg_seconds_per_day <= 0:
        avg_seconds_per_day = 1  # Prevent division by zero

    last_date = df["date"].iloc[-1]
    # Start future dates from the day after the last historical date
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=days_to_predict, freq="D"
    )

    future_seconds = pd.Series([avg_seconds_per_day] * len(future_dates))
    future_df = pd.DataFrame({"date": future_dates, "seconds": future_seconds})

    # Start cumulative seconds from the last cumulative seconds of the historical data
    last_cumulative_seconds = df["cumulative_seconds"].iloc[-1]
    future_df["cumulative_seconds"] = future_seconds.cumsum() + last_cumulative_seconds
    future_df["cumulative_minutes"] = future_df["cumulative_seconds"] / 60
    future_df["cumulative_hours"] = future_df["cumulative_minutes"] / 60

    # Create a single row DataFrame for the last historical point
    last_point = pd.DataFrame(
        {
            "date": [last_date],
            "seconds": [df["seconds"].iloc[-1]],
            "cumulative_seconds": [last_cumulative_seconds],
            "cumulative_minutes": [last_cumulative_seconds / 60],
            "cumulative_hours": [last_cumulative_seconds / 3600],
        }
    )

    # Combine last historical point with future predictions
    combined_df = pd.concat([last_point, future_df], ignore_index=True)

    return combined_df
