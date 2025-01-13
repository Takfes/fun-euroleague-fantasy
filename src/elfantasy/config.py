import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pyprojroot.here import here


class Configuration:
    # directories
    data_dir = here("data")
    data_dir_euroleague_games = Path(data_dir, "euroleague_games")
    data_dir_euroleague_ata = Path(data_dir, "euroleague_data")
    data_dir_datalog = Path(data_dir, "datalog.json")
    data_dir_features_data = Path(data_dir, "euroleague_features")
    data_dir_predictions = Path(data_dir, "euroleague_predictions")
    data_dir_optimizations = Path(data_dir, "euroleague_optimizations")
    # script specific configuration
    # data acquisition
    competition_code = "E"
    season = 2024
    # feature engineering
    limit_up_to_week = 19
    # predicitive modeling
    training_min_weeks = 5
    baseline_column_predictions = "valuation_expanding_mean_plr_tmpr"
    model_column_predictions = "valuation_pred"
    predictive_model_string = "LR"
    predictive_model_id_cols = [
        "week",
        "slug",
        "team_code",
    ]
    predictive_model_cats_cols = ["team_code", "slug"]
    # squad optimization
    optimization_solver = "glpk"
    baseline_column_optimizations = "valuation_expanding_mean_plr_tmpr"  # "valuation_roll_3_plr_tmpr"
    model_column_optimizations = "valuation_pred"
    force_players = []

    def __init__(self, use_dotenv_config_yaml=False):
        if use_dotenv_config_yaml:
            self.configuration_from_yaml(self.resolve_config_yaml_path())
        else:
            self.configuration_from_hardcoded()

    @staticmethod
    def resolve_config_yaml_path():
        """
        Resolves the path to the YAML configuration file.
        This function loads environment variables using `load_dotenv()` and retrieves the path to the YAML configuration
        file from the environment variable `config_yaml`. It then checks if the path is absolute. If not, it attempts to
        resolve the path relative to the script's directory and the project's root directory. If the file is found in either
        location, the function returns the resolved path. If the file is not found in both locations, a `FileNotFoundError`
        is raised.
        Returns:
            str: The resolved path to the YAML configuration file.
        Raises:
            FileNotFoundError: If the YAML configuration file is not found in both the script and project root directories.
        """

        load_dotenv()
        yaml_path = os.getenv("config_yaml")

        # Check if the path is relative
        if not os.path.isabs(yaml_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(script_dir, yaml_path)

            if os.path.exists(potential_path):
                print(f"{yaml_path} found in the current directory. Using : {potential_path}")
                return potential_path

            # If not found, assume it resides at the root directory of the project
            potential_path = here(yaml_path)

            if os.path.exists(potential_path):
                print(f"{yaml_path} found in the project-root directory. Using : {potential_path}")
                return potential_path

            # If not found in both locations, raise an error
            raise FileNotFoundError(
                f"The YAML configuration file was not found in both script and project root directories: {yaml_path}"
            )

    def configuration_from_hardcoded(self):
        # directories
        self.data_dir = Configuration.data_dir
        self.data_dir_euroleague_games = Configuration.data_dir_euroleague_games
        self.data_dir_euroleague_data = Configuration.data_dir_euroleague_ata
        self.data_dir_datalog = Configuration.data_dir_datalog
        self.data_dir_features_data = Configuration.data_dir_features_data
        self.data_dir_predictions = Configuration.data_dir_predictions
        self.data_dir_optimizations = Configuration.data_dir_optimizations
        # script specific configuration
        # data acquisition
        self.competition_code = Configuration.competition_code
        self.season = Configuration.season
        # feature engineering
        self.limit_up_to_week = Configuration.limit_up_to_week
        # predicitive modeling
        self.training_min_weeks = Configuration.training_min_weeks
        self.baseline_column_predictions = Configuration.baseline_column_predictions
        self.model_column_predictions = Configuration.model_column_predictions
        self.predictive_model_string = Configuration.predictive_model_string
        self.predictive_model_id_cols = Configuration.predictive_model_id_cols
        self.predictive_model_cats_cols = Configuration.predictive_model_cats_cols
        # squad optimization
        self.optimization_solver = Configuration.optimization_solver
        self.baseline_column_optimizations = Configuration.baseline_column_optimizations
        self.model_column_optimizations = Configuration.model_column_optimizations
        self.force_players = Configuration.force_players

    def configuration_from_yaml(self, yaml_path):
        with open(yaml_path) as file:
            config = yaml.safe_load(file)

        rootdir = here()
        # directories
        self.data_dir = Path(rootdir / config.get("directories").get("data_dir"))
        self.data_dir_euroleague_games = Path(rootdir / config.get("directories").get("data_dir_euroleague_games"))
        self.data_dir_euroleague_data = Path(rootdir / config.get("directories").get("data_dir_euroleague_data"))
        self.data_dir_datalog = Path(rootdir / config.get("directories").get("data_dir_datalog"))
        self.data_dir_features_data = Path(rootdir / config.get("directories").get("data_dir_features_data"))
        self.data_dir_predictions = Path(rootdir / config.get("directories").get("data_dir_predictions"))
        self.data_dir_optimizations = Path(rootdir / config.get("directories").get("data_dir_optimizations"))
        # script specific configuration
        # data acquisition
        self.competition_code = config.get("data_acquisition").get("competition_code")
        self.season = config.get("data_acquisition").get("season")
        # feature engineering
        self.limit_up_to_week = config.get("feature_engineering").get("limit_up_to_week")
        # predictive modeling
        self.training_min_weeks = config.get("predictive_modeling").get("training_min_weeks")
        self.baseline_column_predictions = config.get("predictive_modeling").get("baseline_column_predictions")
        self.model_column_predictions = config.get("predictive_modeling").get("model_column_predictions")
        self.predictive_model_string = config.get("predictive_modeling").get("predictive_model_string")
        self.predictive_model_id_cols = config.get("predictive_modeling").get("predictive_model_id_cols")
        self.predictive_model_cats_cols = config.get("predictive_modeling").get("predictive_model_cats_cols")
        # squad optimization
        self.optimization_solver = config.get("squad_optimization").get("optimization_solver")
        self.baseline_column_optimizations = config.get("squad_optimization").get("baseline_column_optimizations")
        self.model_column_optimizations = config.get("squad_optimization").get("model_column_optimizations")
        self.force_players = config.get("squad_optimization").get("force_players")
