from dataclasses import dataclass
import pandas as pd


@dataclass
class AnalysisResult:
    """
    Represents the result of data analysis for the Dreaming Spanish application.

    Attributes:
        df (pd.DataFrame): A DataFrame containing the processed data.
        goals_reached (int): The total number of goals reached.
        total_days (int): The total number of days in the dataset.
        current_goal_streak (int): The current streak of consecutive days with goals reached.
        longest_goal_streak (int): The longest streak of consecutive days with goals reached.
    """
    df: pd.DataFrame
    goals_reached: int
    total_days: int
    current_goal_streak: int
    longest_goal_streak: int
