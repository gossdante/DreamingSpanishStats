import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Dreaming Spanish Time Tracker",
    layout="wide"
)

# Data
dates = ["2024/12/27", "2024/12/28", "2024/12/29", "2024/12/30", "2024/12/31", "2025/01/01"]
minutes = [96, 91, 177, 39, 4, 84]

# Create DataFrame
df = pd.DataFrame({
    'date': pd.to_datetime(dates),
    'minutes': minutes
})

# Calculate cumulative minutes
df['cumulative_minutes'] = df['minutes'].cumsum()
df['cumulative_hours'] = df['cumulative_minutes'] / 60

# Calculate average minutes per day
avg_minutes_per_day = df['minutes'].mean()

# Function to predict future values


def generate_future_predictions(df, avg_minutes_per_day, days_to_predict=800):
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                 periods=days_to_predict,
                                 freq='D')

    future_minutes = [avg_minutes_per_day] * len(future_dates)
    future_df = pd.DataFrame({
        'date': future_dates,
        'minutes': future_minutes
    })

    # Calculate cumulative values including historical data
    combined_df = pd.concat([df, future_df])
    combined_df['cumulative_minutes'] = combined_df['minutes'].cumsum()
    combined_df['cumulative_hours'] = combined_df['cumulative_minutes'] / 60

    return combined_df


# Create main containers
st.title("Dreaming Spanish Time Tracker")

# Current stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Minutes Watched", f"{
              df['cumulative_minutes'].iloc[-1]:.0f}")
with col2:
    st.metric("Total Hours Watched", f"{df['cumulative_hours'].iloc[-1]:.1f}")
with col3:
    st.metric("Average Minutes/Day", f"{avg_minutes_per_day:.1f}")

# Generate future predictions
predicted_df = generate_future_predictions(df, avg_minutes_per_day)

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
        # Find the date when milestone will be reached
        milestone_date = predicted_df[predicted_df['cumulative_hours']
                                      >= milestone]['date'].iloc[0]

        # Add horizontal milestone line
        fig_prediction.add_shape(
            type="line",
            x0=df['date'].min(),
            x1=milestone_date,
            y0=milestone,
            y1=milestone,
            line=dict(color=color, dash="dash", width=1)
        )

        # Add milestone label
        fig_prediction.add_annotation(
            x=df['date'].min(),
            y=milestone,
            text=f"{milestone} Hours",
            showarrow=False,
            xshift=-5,
            xanchor='right',
            font=dict(color=color)
        )

        # Add date annotation
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

fig_prediction.update_layout(
    title='Projected Growth and Milestones',
    xaxis_title='Date',
    yaxis_title='Cumulative Hours',
    showlegend=True,
    height=600  # Make the graph taller
)

# Update axes ranges
fig_prediction.update_yaxes(
    range=[0, min(predicted_df['cumulative_hours'].max() * 1.1, 1600)])

st.plotly_chart(fig_prediction, use_container_width=True)

# Text predictions
current_hours = df['cumulative_hours'].iloc[-1]
col1, col2 = st.columns(2)

with col1:
    st.subheader("Expected Milestone Dates")
    for milestone in milestones:
        if current_hours < milestone:
            days_to_milestone = ((milestone - current_hours)
                                 * 60) / avg_minutes_per_day
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

# Daily breakdown
st.subheader("Daily Watching Time")
daily_fig = px.bar(df,
                   x='date',
                   y='minutes',
                   title='Daily Minutes Watched',
                   labels={'minutes': 'Minutes', 'date': 'Date'})
st.plotly_chart(daily_fig, use_container_width=True)

# Additional insights
st.subheader("Additional Insights")
col1, col2 = st.columns(2)

with col1:
    st.metric("Highest Daily Watch Time",
              f"{df['minutes'].max()} minutes",
              f"on {df.loc[df['minutes'].idxmax(), 'date'].strftime('%Y-%m-%d')}")

with col2:
    next_milestone = next(m for m in milestones if m > current_hours)
    days_to_next = ((next_milestone - current_hours) * 60) / \
        avg_minutes_per_day
    st.metric(f"Days to {next_milestone} Hours",
              f"{days_to_next:.1f} days")

# Add date range for context
st.caption(f"Data range: {df['date'].min().strftime(
    '%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
