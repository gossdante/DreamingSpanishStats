import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
import json

# Function to fetch data from Dreaming Spanish API


def fetch_ds_data(token):
    url = "https://www.dreamingspanish.com/.netlify/functions/dayWatchedTime"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def load_data(token):
    """Fetch and process data from API"""
    api_data = fetch_ds_data(token)
    if not api_data:
        return None

    # Convert API data to DataFrame
    df = pd.DataFrame(api_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Add goal tracking metrics
    total_days = len(df)
    goals_reached = df['goalReached'].sum()
    goal_streak = (df['goalReached'] == True).astype(int)

    # Calculate current goal streak
    df['goal_streak_group'] = (goal_streak != goal_streak.shift()).cumsum()
    df['current_goal_streak'] = df.groupby('goal_streak_group')[
        'goalReached'].cumsum()
    current_goal_streak = df['current_goal_streak'].iloc[-1] if df['goalReached'].iloc[-1] else 0

    # Calculate longest goal streak
    goal_streak_lengths = df[df['goalReached'] ==
                             True].groupby('goal_streak_group').size()
    longest_goal_streak = goal_streak_lengths.max(
    ) if not goal_streak_lengths.empty else 0

    return df, goals_reached, total_days, current_goal_streak, longest_goal_streak


# Set page config
st.set_page_config(
    page_title="Dreaming Spanish Time Tracker",
    layout="wide"
)

# Create main containers
st.title("Dreaming Spanish Time Tracker")
st.subheader("Analyze your viewing habits and set goals")

# Add token input and button in an aligned row
st.write("")  # Add some spacing
col1, col2 = st.columns([4, 1])
with col1:
    token = st.text_input("Enter your bearer token:", type="password",
                          key="token_input", label_visibility="collapsed")
with col2:
    go_button = st.button("Go", type="primary", use_container_width=True)

if not token:
    st.warning("Please enter your bearer token to fetch data")
    st.stop()

# Load data when token is provided and button is clicked
if 'data' not in st.session_state or go_button:
    with st.spinner('Fetching data...'):
        data = load_data(token)
        if data is None:
            st.error("Failed to fetch data")
            st.stop()
        st.session_state.data = data

# Unpack data from session state
df, goals_reached, total_days, current_goal_streak, longest_goal_streak = st.session_state.data

# Create DataFrame with seconds data
seconds = df['timeSeconds'].tolist()
dates = df['date'].dt.strftime('%Y/%m/%d').tolist()

df = pd.DataFrame({
    'date': pd.to_datetime(dates),
    'seconds': seconds
})

# Calculate cumulative seconds and streak
df['cumulative_seconds'] = df['seconds'].cumsum()
df['cumulative_minutes'] = df['cumulative_seconds'] / 60
df['cumulative_hours'] = df['cumulative_minutes'] / 60
df['streak'] = (df['seconds'] > 0).astype(int)

# Calculate current streak
df['streak_group'] = (df['streak'] != df['streak'].shift()).cumsum()
df['current_streak'] = df.groupby('streak_group')['streak'].cumsum()
current_streak = df['current_streak'].iloc[-1] if df['streak'].iloc[-1] == 1 else 0

# Calculate all-time longest streak
streak_lengths = df[df['streak'] == 1].groupby('streak_group').size()
longest_streak = streak_lengths.max() if not streak_lengths.empty else 0

# Calculate moving averages
df['7day_avg'] = df['seconds'].rolling(7, min_periods=1).mean()
df['30day_avg'] = df['seconds'].rolling(30, min_periods=1).mean()

# Calculate average seconds per day
avg_seconds_per_day = df['seconds'].mean()

# Function to predict future values


def generate_future_predictions(df, avg_seconds_per_day, days_to_predict=800):
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                 periods=days_to_predict,
                                 freq='D')

    future_seconds = [avg_seconds_per_day] * len(future_dates)
    future_df = pd.DataFrame({
        'date': future_dates,
        'seconds': future_seconds
    })

    combined_df = pd.concat([df, future_df])
    combined_df['cumulative_seconds'] = combined_df['seconds'].cumsum()
    combined_df['cumulative_minutes'] = combined_df['cumulative_seconds'] / 60
    combined_df['cumulative_hours'] = combined_df['cumulative_minutes'] / 60

    return combined_df


# Current stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Hours Watched", f"{df['cumulative_hours'].iloc[-1]:.1f}")
with col2:
    st.metric("Average Minutes/Day", f"{(avg_seconds_per_day / 60):.1f}")
with col3:
    st.metric("Current Streak", f"{current_streak} days")
with col4:
    st.metric("Longest Streak", f"{longest_streak} days")

# Generate future predictions
predicted_df = generate_future_predictions(df, avg_seconds_per_day)

# Create milestone prediction visualization
fig_prediction = go.Figure()

# Add historical data
fig_prediction.add_trace(go.Scatter(
    x=df['date'],
    y=df['cumulative_hours'],
    name='Historical Data',
    line=dict(color='blue'),
    mode='lines+markers'
))

# Add predicted data
fig_prediction.add_trace(go.Scatter(
    x=predicted_df['date'][len(df):],
    y=predicted_df['cumulative_hours'][len(df):],
    name='Predicted Growth',
    line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash'),
    mode='lines'
))

# Add milestone lines and annotations
milestones = [50, 150, 300, 600, 1000, 1500]
colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']

for milestone, color in zip(milestones, colors):
    if milestone <= predicted_df['cumulative_hours'].max():
        milestone_date = predicted_df[predicted_df['cumulative_hours']
                                      >= milestone]['date'].iloc[0]

        fig_prediction.add_shape(
            type="line",
            x0=df['date'].min(),
            x1=milestone_date,
            y0=milestone,
            y1=milestone,
            line=dict(color=color, dash="dash", width=1)
        )

        fig_prediction.add_annotation(
            x=df['date'].min(),
            y=milestone,
            text=f"{milestone} Hours",
            showarrow=False,
            xshift=-5,
            xanchor='right',
            font=dict(color=color)
        )

        fig_prediction.add_annotation(
            x=milestone_date,
            y=milestone,
            text=milestone_date.strftime('%Y-%m-%d'),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=color,
            font=dict(color=color, size=10),
            xanchor='left',
            yanchor='bottom'
        )

# Find the next 3 upcoming milestones and their dates
current_hours = df['cumulative_hours'].iloc[-1]
upcoming_milestones = [m for m in milestones if m > current_hours][:3]
y_axis_max = upcoming_milestones[2] if len(
    upcoming_milestones) >= 3 else milestones[-1]

# Get the date for the third upcoming milestone (or last milestone if less than 3 remain)
target_milestone = upcoming_milestones[2] if len(
    upcoming_milestones) >= 3 else upcoming_milestones[-1]
x_axis_max_date = predicted_df[predicted_df['cumulative_hours']
                               >= target_milestone]['date'].iloc[0]

fig_prediction.update_layout(
    title='Projected Growth and Milestones',
    xaxis_title='Date',
    yaxis_title='Cumulative Hours',
    showlegend=True,
    height=600,
    yaxis=dict(
        range=[0, y_axis_max * 1.35]  # Increase vertical padding to 35%
    ),
    xaxis=dict(
        # Add 15 days padding to start and end dates
        range=[df['date'].min() - timedelta(days=15),
               x_axis_max_date + timedelta(days=15)]
    )
)

st.plotly_chart(fig_prediction, use_container_width=True)

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(
    ["Daily Breakdown", "Moving Averages", "Weekly Heatmap"])

with tab1:
    # Daily breakdown
    daily_fig = px.bar(df,
                       x='date',
                       y=df['seconds'] / 60,  # Convert to minutes for display
                       title='Daily Minutes Watched',
                       labels={'value': 'Minutes', 'date': 'Date'})
    st.plotly_chart(daily_fig, use_container_width=True)

with tab2:
    # Moving averages visualization
    moving_avg_fig = go.Figure()

    moving_avg_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['seconds'] / 60,  # Convert to minutes
        name='Daily Minutes',
        mode='markers',
        marker=dict(size=6)
    ))

    moving_avg_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['7day_avg'] / 60,  # Convert to minutes
        name='7-day Average',
        line=dict(color='orange')
    ))

    moving_avg_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['30day_avg'] / 60,  # Convert to minutes
        name='30-day Average',
        line=dict(color='green')
    ))

    moving_avg_fig.update_layout(
        title='Daily Minutes with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Minutes',
        height=400
    )

    st.plotly_chart(moving_avg_fig, use_container_width=True)

with tab3:
    # Weekly heatmap
    df['weekday'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week

    heatmap_fig = px.density_heatmap(
        df,
        x='weekday',
        y='week',
        z=df['seconds'] / 60,  # Convert to minutes
        title='Weekly Viewing Pattern',
        labels={'value': 'Minutes Watched',
                'weekday': 'Day of Week', 'week': 'Week Number'},
        category_orders={'weekday': [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
    )

    heatmap_fig.update_layout(height=400)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Text predictions
current_hours = df['cumulative_hours'].iloc[-1]
col1, col2 = st.columns(2)

with col1:
    st.subheader("Expected Milestone Dates")
    for milestone in milestones:
        if current_hours < milestone:
            days_to_milestone = ((milestone - current_hours)
                                 * 3600) / avg_seconds_per_day
            predicted_date = df['date'].iloc[-1] + \
                timedelta(days=days_to_milestone)
            st.write(f"📅 {milestone} hours: {predicted_date.strftime(
                '%Y-%m-%d')} ({days_to_milestone:.0f} days)")
        else:
            st.write(f"✅ {milestone} hours: Already achieved!")

with col2:
    st.subheader("Progress Overview")
    for milestone in milestones:
        if current_hours < milestone:
            percentage = (current_hours / milestone) * 100
            st.write(f"Progress to {milestone} hours: {percentage:.1f}%")
            st.progress(percentage / 100)

# Additional insights
st.subheader("Additional Insights")
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Best day stats
    best_day_idx = df['seconds'].idxmax()
    best_day = df.loc[best_day_idx]
    st.metric(
        "Best Day",
        f"{(best_day['seconds'] / 60):.0f} min",
        f"{best_day['date'].strftime('%a %b %d')}"
    )
    # Add consistency metric
    days_watched = (df['seconds'] > 0).sum()
    consistency = (days_watched / len(df)) * 100
    st.metric("Consistency", f"{consistency:.1f}%",
              f"{days_watched} of {len(df)} days")

with col2:
    # Streak information
    st.metric("Current Streak", f"{current_streak} days")
    avg_streak = streak_lengths.mean() if not streak_lengths.empty else 0
    st.metric("Average Streak", f"{avg_streak:.1f} days",
              f"Best: {longest_streak} days")

with col3:
    # Time comparisons
    last_7_total = df.tail(7)['seconds'].sum()
    previous_7_total = df.iloc[-14:-7]['seconds'].sum() if len(df) >= 14 else 0
    week_change = last_7_total - previous_7_total
    st.metric("Last 7 Days Total", f"{(last_7_total / 60):.0f} min",
              f"{(week_change / 60):+.0f} min vs previous week")

    weekly_avg = df.tail(7)['seconds'].mean()
    st.metric("7-Day Average", f"{(weekly_avg / 60):.1f} min/day",
              f"{((weekly_avg - avg_seconds_per_day) / 60):+.1f} vs overall")

with col4:
    # Achievement metrics
    total_time = df['seconds'].sum()
    milestone_count = sum(m <= df['cumulative_hours'].iloc[-1]
                          for m in milestones)
    st.metric("Total Time", f"{(total_time / 60):.0f} min",
              f"{milestone_count} milestones reached")

    goal_rate = (goals_reached / total_days) * 100
    st.metric("Goal Achievement", f"{goals_reached} days",
              f"{goal_rate:.1f}% of days")

    st.metric("Goal Streak", f"{current_goal_streak} days",
              f"Best: {longest_goal_streak} days")

# Add date range for context
st.caption(f"Data range: {df['date'].min().strftime(
    '%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
