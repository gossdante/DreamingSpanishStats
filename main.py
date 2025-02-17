"""
A Streamlit dashboard application for tracking and visualizing Dreaming Spanish learning progress.

This application provides an interactive interface to:
- Load and display viewing data from the Dreaming Spanish API
- Visualize viewing patterns
- Track progress towards learning milestones (50, 150, 300, 600, 1000, 1500 hours)
- Generate predictions for future milestone achievements
- Display statistics like streaks, best days, and goal completion rates
- Export viewing data to CSV format
"""

from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import os

from src.utils import generate_future_predictions, load_data

# Set pandas option for future compatibility
pd.set_option("future.no_silent_downcasting", True)


MILESTONES = [50, 150, 300, 600, 1000, 1500]
COLOUR_PALETTE = {
    "primary": "#2E86C1",  # Primary blue
    "7day_avg": "#FFA500",  # Orange
    "30day_avg": "#2ECC71",  # Green
    # Milestone colors - using an accessible and distinguishable gradient
    "50": "#FF6B6B",  # Coral red
    "150": "#4ECDC4",  # Turquoise
    "300": "#9B59B6",  # Purple
    "600": "#F1C40F",  # Yellow
    "1000": "#E67E22",  # Orange
    "1500": "#2ECC71",  # Green
}

st.set_page_config(page_title="Dreaming Spanish Time Tracker", layout="wide")

st.title("Dreaming Spanish Time Tracker")
st.subheader("Analyze your viewing habits and predict your progress")
st.info("This tool is new and may contain bugs and slight statistical errors for the time being. Please report any issues on the GitHub repository. Thank you!")

button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
with button_col1:
    st.link_button("☁️ Official Progress",
                   "https://www.dreamingspanish.com/progress", use_container_width=True)

with button_col2:
    st.link_button(
        "🪲 Report Issue", "http://github.com/HarryPeach/dreamingspanishstats/issues", use_container_width=True)

with button_col3:
    st.link_button(
        "📖 Source Code", "http://github.com/HarryPeach/dreamingspanishstats", use_container_width=True)

# Add token input and buttons in an aligned row
st.write("")  # Add some spacing
col1, col2 = st.columns([4, 1])
with col1:
    token = st.text_input(
        "Enter your bearer token:",
        type="password",
        key="token_input",
        label_visibility="collapsed",
    )
with col2:
    go_button = st.button("Go", type="primary", use_container_width=True)

load_dotenv()
token = os.getenv("BEARER_TOKEN")
# Check if token is provided

if not token:
    st.warning("Please enter your bearer token above to fetch data")
    # Load and display README
    try:
        with open("bearer_how_to.md", "r") as file:
            readme_content = file.read()
            if readme_content:
                st.markdown(readme_content, unsafe_allow_html=True)
    except FileNotFoundError:
        pass
    st.stop()

# Load data when token is provided and button is clicked
if "data" not in st.session_state or go_button:
    with st.spinner("Fetching data..."):
        data = load_data(token)
        if data is None:
            st.error("Failed to fetch data from the DreamingSpnaish API. Please check your bearer token, ensuring it doesn't contain anything extra such as 'token:' at the beginning.")
            st.stop()
        st.session_state.data = data


result = st.session_state.data
df = result.df
goals_reached = result.goals_reached
total_days = result.total_days
current_goal_streak = result.current_goal_streak
longest_goal_streak = result.longest_goal_streak

# Just rename the column instead of recreating the DataFrame
df = df.rename(columns={"timeSeconds": "seconds"})

# Calculate cumulative seconds and streak
seconds = df["seconds"].tolist()
dates = df["date"].dt.strftime("%Y/%m/%d").tolist()

df = pd.DataFrame({"date": pd.to_datetime(dates), "seconds": seconds})

# Calculate cumulative seconds and streak
df["cumulative_seconds"] = df["seconds"].cumsum() + (50 * 60 * 60)
df["cumulative_minutes"] = df["cumulative_seconds"] / 60
df["cumulative_hours"] = df["cumulative_minutes"] / 60
df["streak"] = (df["seconds"] > 0).astype(int)

# Calculate current streak
df["streak_group"] = (df["streak"] != df["streak"].shift()).cumsum()
df["current_streak"] = df.groupby("streak_group")["streak"].cumsum()
current_streak = df["current_streak"].iloc[-1] if df["streak"].iloc[-1] == 1 else 0

# Calculate all-time longest streak
streak_lengths = df[df["streak"] == 1].groupby("streak_group").size()
longest_streak = streak_lengths.max() if not streak_lengths.empty else 0

# Calculate moving averages
df["7day_avg"] = df["seconds"].rolling(7, min_periods=1).mean()
df["30day_avg"] = df["seconds"].rolling(30, min_periods=1).mean()

avg_seconds_per_day = df["seconds"].mean()

with st.container(border=True):
    st.subheader("Basic Stats")

    # Current stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hours Watched",
                  f"{df['cumulative_hours'].iloc[-1]:.1f}")
    with col2:
        st.metric("Average Minutes/Day", f"{(avg_seconds_per_day / 60):.1f}")
    with col3:
        st.metric("Current Streak", f"{current_streak} days")
    with col4:
        st.metric("Longest Streak", f"{longest_streak} days")

with st.container(border=True):
    st.subheader("Averages")

    # Total, 30 Day, and 7 Day averages
    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric("Total Average", f"{(avg_seconds_per_day / 60):.1f} min/day")
    with col7:
        st.metric("30-Day Average",
                  f"{(df['30day_avg'].iloc[-1] / 60):.1f} min/day")
    with col8:
        st.metric("7-Day Average",
                  f"{(df['7day_avg'].iloc[-1] / 60):.1f} min/day")

with st.container(border=True):
    st.subheader("Projected Growth")

    # Calculate target milestone
    current_hours = df["cumulative_hours"].iloc[-1]
    upcoming_milestones = [m for m in MILESTONES if m > current_hours][:3]
    target_milestone = upcoming_milestones[2] if len(
        upcoming_milestones) >= 3 else MILESTONES[-1]

    # Calculate current moving averages for predictions
    current_7day_avg = df["7day_avg"].iloc[-1]
    current_30day_avg = df["30day_avg"].iloc[-1]

    # Generate predictions up to target milestone
    predicted_df = generate_future_predictions(
        df, avg_seconds_per_day, target_milestone)
    predicted_df_7day = generate_future_predictions(
        df, current_7day_avg, target_milestone)
    predicted_df_30day = generate_future_predictions(
        df, current_30day_avg, target_milestone)

    # Create milestone prediction visualization
    fig_prediction = go.Figure()

    # Add historical data
    fig_prediction.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_hours"],
            name="Historical Data",
            line=dict(color=COLOUR_PALETTE["primary"]),
            mode="lines+markers",
        )
    )

    # Add predicted data - Overall Average
    fig_prediction.add_trace(
        go.Scatter(
            x=predicted_df["date"],
            y=predicted_df["cumulative_hours"],
            name="Predicted (Overall Avg)",
            line=dict(color=f"{COLOUR_PALETTE['primary']}", dash="dash"),
            mode="lines",
            opacity=0.5,
        )
    )

    # Add predicted data - 7-Day Average
    fig_prediction.add_trace(
        go.Scatter(
            x=predicted_df_7day["date"],
            y=predicted_df_7day["cumulative_hours"],
            name="Predicted (7-Day Avg)",
            line=dict(color=COLOUR_PALETTE["7day_avg"], dash="dot"),
            mode="lines",
            opacity=0.5,
        )
    )

    # Add predicted data - 30-Day Average
    fig_prediction.add_trace(
        go.Scatter(
            x=predicted_df_30day["date"],
            y=predicted_df_30day["cumulative_hours"],
            name="Predicted (30-Day Avg)",
            line=dict(color=COLOUR_PALETTE["30day_avg"], dash="dot"),
            mode="lines",
            opacity=0.5,
        )
    )

    for milestone in MILESTONES:
        color = COLOUR_PALETTE[str(milestone)]
        if milestone <= predicted_df["cumulative_hours"].max():
            milestone_date = predicted_df[
                predicted_df["cumulative_hours"] >= milestone
            ]["date"].iloc[0]

            fig_prediction.add_shape(
                type="line",
                x0=df["date"].min(),
                x1=milestone_date,
                y0=milestone,
                y1=milestone,
                line=dict(color=color, dash="dash", width=1),
            )

            fig_prediction.add_annotation(
                x=df["date"].min(),
                y=milestone,
                text=f"{milestone} Hours",
                showarrow=False,
                xshift=-5,
                xanchor="right",
                font=dict(color=color),
            )

            fig_prediction.add_annotation(
                x=milestone_date,
                y=milestone,
                text=milestone_date.strftime("%Y-%m-%d"),
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor=color,
                font=dict(color=color, size=10),
                xanchor="left",
                yanchor="bottom",
            )

    # Find the next 3 upcoming milestones and their dates
    current_hours = df["cumulative_hours"].iloc[-1]
    upcoming_milestones = [m for m in MILESTONES if m > current_hours][:3]
    y_axis_max = (
        upcoming_milestones[2] if len(
            upcoming_milestones) >= 3 else MILESTONES[-1]
    )

    # Get the date for the third upcoming milestone (or last milestone if less than 3 remain)
    if len(upcoming_milestones) > 0:
        target_milestone = upcoming_milestones[min(
            2, len(upcoming_milestones) - 1)]
        milestone_data = predicted_df[
            predicted_df["cumulative_hours"] >= target_milestone
        ]
        if len(milestone_data) > 0:
            x_axis_max_date = milestone_data["date"].iloc[0]
        else:
            x_axis_max_date = predicted_df["date"].max()
    else:
        x_axis_max_date = predicted_df["date"].max()

    fig_prediction.update_layout(
        # title='Projected Growth and Milestones',
        xaxis_title="Date",
        yaxis_title="Cumulative Hours",
        showlegend=True,
        height=600,
        margin=dict(l=20, r=20, t=10, b=0),
        yaxis=dict(
            autorange=True,
        ),
        xaxis=dict(
            autorange=True,
        ),
    )

    st.plotly_chart(fig_prediction, use_container_width=True)

with st.container(border=True):
    st.subheader("Additional Graphs")
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(
        ["Daily Breakdown", "Moving Averages", "Yearly Heatmap"])

    with tab1:
        # Daily breakdown
        daily_fig = px.bar(
            df,
            x="date",
            y=df["seconds"] / 60,  # Convert to minutes for display
            title="Daily Minutes Watched",
            labels={"value": "Minutes", "date": "Date"},
        )
        daily_fig.update_yaxes(
            dtick=15, title="Minutes Watched", ticklabelstep=2)
        st.plotly_chart(daily_fig, use_container_width=True)

    with tab2:
        # Moving averages visualization
        moving_avg_fig = go.Figure()

        moving_avg_fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["seconds"] / 60,  # Convert to minutes
                name="Daily Minutes",
                mode="markers",
                marker=dict(size=6),
            )
        )

        moving_avg_fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["7day_avg"] / 60,  # Convert to minutes
                name="7-day Average",
                line=dict(color=COLOUR_PALETTE["7day_avg"]),
            )
        )

        moving_avg_fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["30day_avg"] / 60,  # Convert to minutes
                name="30-day Average",
                line=dict(color=COLOUR_PALETTE["30day_avg"]),
            )
        )

        moving_avg_fig.update_layout(
            title="Daily Minutes with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Minutes",
            height=400,
        )

        moving_avg_fig.update_yaxes(
            dtick=15, title="Minutes Watched", ticklabelstep=2)

        st.plotly_chart(moving_avg_fig, use_container_width=True)

    with tab3:
        # Create a complete year date range
        today = pd.Timestamp.now()
        year_start = pd.Timestamp(today.year, 1, 1)
        year_end = pd.Timestamp(today.year, 12, 31)
        all_dates = pd.date_range(year_start, year_end, freq="D")

        # Create a DataFrame with all dates
        full_year_df = pd.DataFrame({"date": all_dates})
        full_year_df["seconds"] = 0

        # Merge with actual data
        full_year_df = full_year_df.merge(
            df[["date", "seconds"]], on="date", how="left"
        )
        full_year_df["seconds"] = full_year_df["seconds_y"].fillna(0)

        # Calculate week and weekday using isocalendar
        isocalendar_df = full_year_df["date"].dt.isocalendar()
        full_year_df["weekday"] = isocalendar_df["day"]

        # Handle week numbers correctly
        full_year_df["week"] = isocalendar_df["week"]
        # Adjust week numbers for consistency
        mask = (full_year_df["date"].dt.month == 12) & (
            full_year_df["week"] <= 1)
        full_year_df.loc[mask, "week"] = full_year_df.loc[mask, "week"] + 52
        mask = (full_year_df["date"].dt.month == 1) & (
            full_year_df["week"] >= 52)
        full_year_df.loc[mask, "week"] = full_year_df.loc[mask, "week"] - 52

        # Rest of the heatmap code remains the same
        heatmap_fig = go.Figure()

        heatmap_fig.add_trace(
            go.Heatmap(
                x=full_year_df["week"],
                y=full_year_df["weekday"],
                z=full_year_df["seconds"] / 60,  # Convert to minutes
                colorscale=[
                    [0, "rgba(227,224,227,.5)"],  # Grey for zeros/future
                    [0.001, "rgb(243,231,154)"],
                    [0.5, "rgb(246,90,109)"],
                    [1, "rgb(126,29,103)"],
                ],
                showscale=True,
                colorbar=dict(title="Minutes"),
                hoverongaps=False,
                hovertemplate="Date: %{customdata}<br>"
                + "Minutes: %{z:.1f}<extra></extra>",
                customdata=full_year_df["date"].dt.strftime("%Y-%m-%d"),
                xgap=3,  # Add 3 pixels gap between columns
                ygap=3,  # Add 3 pixels gap between rows
            )
        )

        # Update layout for GitHub-style appearance
        heatmap_fig.update_layout(
            title="Yearly Activity Heatmap",
            xaxis_title="Week",
            yaxis_title="Day of Week",
            height=300,
            yaxis=dict(
                ticktext=["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                tickvals=[0, 1, 2, 3, 4, 5, 6, 7],
                showgrid=False,
                autorange="reversed",  # This ensures Mon-Sun order
            ),
            xaxis=dict(
                showgrid=False,
                dtick=1,  # Show all week numbers
                range=[0.5, 53.5],  # Fix the range to show all weeks
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(heatmap_fig, use_container_width=True)

with st.container(border=True):
    # Text predictions
    current_hours = df["cumulative_hours"].iloc[-1]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Expected Milestone Dates")
        for milestone in MILESTONES:
            if current_hours < milestone:
                days_to_milestone = (
                    (milestone - current_hours) * 3600
                ) / avg_seconds_per_day
                predicted_date = df["date"].iloc[-1] + \
                    timedelta(days=days_to_milestone)
                st.write(
                f"📅 {milestone} hours: {predicted_date.strftime('%Y-%m-%d')} "
                f"({days_to_milestone:.0f} days)"
            )
            else:
                st.write(f"✅ {milestone} hours: Already achieved!")

    with col2:
        st.subheader("Progress Overview")
        for milestone in MILESTONES:
            if current_hours < milestone:
                percentage = (current_hours / milestone) * 100
                st.write(f"Progress to {milestone} hours: {percentage:.1f}%")
                st.progress(percentage / 100)

with st.container(border=True):
    st.subheader("Additional Insights")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Best day stats
        best_day_idx = df["seconds"].idxmax()
        best_day = df.loc[best_day_idx]
        st.metric(
            "Best Day",
            f"{(best_day['seconds'] / 60):.0f} min",
            f"{best_day['date'].strftime('%a %b %d')}",
        )
        # Add consistency metric
        days_watched = (df["seconds"] > 0).sum()
        consistency = (days_watched / len(df)) * 100
        st.metric(
            "Consistency", f"{consistency:.1f}%", f"{days_watched} of {len(df)} days"
        )

    with col2:
        # Streak information
        st.metric("Current Streak", f"{current_streak} days")
        avg_streak = streak_lengths.mean() if not streak_lengths.empty else 0
        st.metric(
            "Average Streak", f"{avg_streak:.1f} days", f"Best: {longest_streak} days"
        )

        st.metric(
            "Goal Streak",
            f"{current_goal_streak} days",
            f"Best: {longest_goal_streak} days",
        )

    with col3:
        # Time comparisons
        last_7_total = df.tail(7)["seconds"].sum()
        previous_7_total = df.iloc[-14:-
                                   7]["seconds"].sum() if len(df) >= 14 else 0
        week_change = last_7_total - previous_7_total
        st.metric(
            "Last 7 Days Total",
            f"{(last_7_total / 60):.0f} min",
            f"{(week_change / 60):+.0f} min vs previous week",
        )

        weekly_avg = df.tail(7)["seconds"].mean()
        st.metric(
            "7-Day Average",
            f"{(weekly_avg / 60):.1f} min/day",
            f"{((weekly_avg - avg_seconds_per_day) / 60):+.1f} vs overall",
        )

    with col4:
        # Achievement metrics
        total_time = df["seconds"].sum()
        milestone_count = sum(
            m <= df["cumulative_hours"].iloc[-1] for m in MILESTONES)
        st.metric(
            "Total Time",
            f"{(total_time / 60):.0f} min",
            f"{milestone_count} milestones reached",
            delta_color="off",
        )

        goal_rate = (goals_reached / total_days) * 100
        st.metric(
            "Goal Achievement", f"{goals_reached} days", f"{goal_rate:.1f}% of days"
        )


with st.container(border=True):
    st.subheader("Additional Tools")
    result = st.session_state.data
    st.download_button(
        label="📥 Export Data to CSV",
        data=result.df.to_csv(index=False),
        file_name="dreaming_spanish_data.csv",
        mime="text/csv",
    )

# Add date range for context
st.caption(
    f"Data range: {df['date'].min().strftime('%Y-%m-%d')} to "
    f"{df['date'].max().strftime('%Y-%m-%d')}"
)

