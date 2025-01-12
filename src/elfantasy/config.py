from pathlib import Path

from pyprojroot.here import here


class Configuration:
    # directories
    DATA_DIR = here("data")
    DATA_DIR_EUROLEAGUE_GAMES = Path(DATA_DIR, "euroleague_games")
    DATA_DIR_EUROLEAGUE_DATA = Path(DATA_DIR, "euroleague_data")
    DATA_DIR_DATALOG = Path(DATA_DIR, "datalog.json")
    DATA_DIR_FEATURES_DATA = Path(DATA_DIR, "euroleague_features")
    DATA_DIR_PREDICTIONS = Path(DATA_DIR, "euroleague_predictions")
    DATA_DIR_OPTIMIZATIONS = Path(DATA_DIR, "euroleague_optimizations")

    # configuration
    competition_code = "E"
    season = 2024
    limit_up_to_week = 19
    training_min_weeks = 5
    baseline_column = "valuation_roll_3_plr_tmpr"
    prediction_column = "valuation_pred"

    def __init__(self):
        # generic configuration
        self.competition_code = Configuration.competition_code
        self.season = Configuration.season
        self.limit_up_to_week = Configuration.limit_up_to_week
        self.training_min_weeks = Configuration.training_min_weeks
        self.baseline_column = Configuration.baseline_column
        self.prediction_column = Configuration.prediction_column
        # directories
        self.data_dir = Configuration.DATA_DIR
        self.data_dir_euroleague_games = Configuration.DATA_DIR_EUROLEAGUE_GAMES
        self.data_dir_euroleague_data = Configuration.DATA_DIR_EUROLEAGUE_DATA
        self.data_dir_datalog = Configuration.DATA_DIR_DATALOG
        self.data_dir_features_data = Configuration.DATA_DIR_FEATURES_DATA
        self.data_dir_predictions = Configuration.DATA_DIR_PREDICTIONS
        self.data_dir_optimizations = Configuration.DATA_DIR_OPTIMIZATIONS
