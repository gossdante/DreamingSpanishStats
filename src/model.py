from dataclasses import dataclass
import pandas as pd


@dataclass
class AnalysisResult:
    df: pd.DataFrame
    goals_reached: int
    total_days: int
    current_goal_streak: int
    longest_goal_streak: int
