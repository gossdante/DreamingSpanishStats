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
minutes = [96, 91, 177, 39, 4, 84]  # Your minutes list
dates = pd.date_range(end=datetime.now().date(), periods=len(
    minutes)).strftime('%Y/%m/%d').tolist()

# Create DataFrame
df = pd.DataFrame({
    'date': pd.to_datetime(dates),
    'minutes': minutes
})

# Calculate cumulative minutes and streak
df['cumulative_minutes'] = df['minutes'].cumsum()
df['cumulative_hours'] = df['cumulative_minutes'] / 60
df['streak'] = (df['minutes'] > 0).astype(int)

# Calculate current streak
df['streak_group'] = (df['streak'] != df['streak'].shift()).cumsum()
df['current_streak'] = df.groupby('streak_group')['streak'].cumsum()
current_streak = df['current_streak'].iloc[-1] if df['streak'].iloc[-1] == 1 else 0

# Calculate all-time longest streak
streak_lengths = df[df['streak'] == 1].groupby('streak_group').size()
longest_streak = streak_lengths.max() if not streak_lengths.empty else 0

# Calculate moving averages
df['7day_avg'] = df['minutes'].rolling(7, min_periods=1).mean()
df['30day_avg'] = df['minutes'].rolling(30, min_periods=1).mean()

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

    combined_df = pd.concat([df, future_df])
    combined_df['cumulative_minutes'] = combined_df['minutes'].cumsum()
    combined_df['cumulative_hours'] = combined_df['cumulative_minutes'] / 60

    return combined_df


# Create main containers
st.title("Dreaming Spanish Time Tracker")

# Current stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Hours Watched", f"{df['cumulative_hours'].iloc[-1]:.1f}")
with col2:
    st.metric("Average Minutes/Day", f"{avg_minutes_per_day:.1f}")
with col3:
    st.metric("Current Streak", f"{current_streak} days")
with col4:
    st.metric("Longest Streak", f"{longest_streak} days")

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
                       y='minutes',
                       title='Daily Minutes Watched',
                       labels={'minutes': 'Minutes', 'date': 'Date'})
    st.plotly_chart(daily_fig, use_container_width=True)

with tab2:
    # Moving averages visualization
    moving_avg_fig = go.Figure()

    moving_avg_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['minutes'],
        name='Daily Minutes',
        mode='markers',
        marker=dict(size=6)
    ))

    moving_avg_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['7day_avg'],
        name='7-day Average',
        line=dict(color='orange')
    ))

    moving_avg_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['30day_avg'],
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
        z='minutes',
        title='Weekly Viewing Pattern',
        labels={'weekday': 'Day of Week', 'week': 'Week Number',
                'minutes': 'Minutes Watched'},
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

# Additional insights
st.subheader("Additional Insights")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Highest Daily Watch Time",
              f"{df['minutes'].max()} minutes",
              f"on {df.loc[df['minutes'].idxmax(), 'date'].strftime('%Y-%m-%d')}")

with col2:
    next_milestone = next(m for m in milestones if m > current_hours)
    days_to_next = ((next_milestone - current_hours) * 60) / \
        avg_minutes_per_day
    st.metric(f"Days to {next_milestone} Hours", f"{days_to_next:.1f} days")

with col3:
    weekly_avg = df.tail(7)['minutes'].mean()
    st.metric("Last 7 Days Average",
              f"{weekly_avg:.1f} minutes",
              f"{weekly_avg - avg_minutes_per_day:+.1f} vs overall average")

# Add date range for context
st.caption(f"Data range: {df['date'].min().strftime(
    '%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
