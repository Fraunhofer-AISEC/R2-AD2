from pathlib import Path
from argparse import ArgumentParser

# The path the project resides in
BASE_PATH = Path(__file__).parent.parent

# Alarm net dimensions
ALARM_HUGE = (5000, 2000, 500, 200)
ALARM_BIG = (1000, 500, 200, 75)
ALARM_SMALL = (100, 50, 25, 10)

# Evaluation intervals
N_TRAIN_ANOMALIES = [5, 10, 25, 50, 100]
P_POLLUTIONS = [0.0, 0.025, 0.05, 0.075, 0.1]


# Standard arguments
def add_standard_arguments(parser: ArgumentParser):
    """
    Add basic arguments useful to all experiments
    :param parser: argument parser object
    :return: argument parser with standard arguments added
    """

    parser.add_argument(
        "random_seed", type=int, help="Seed to fix randomness"
    )
    parser.add_argument(
        "data_split", type=str, help="Data split on which to evaluate the performance (i.e. val or test)"
    )
    parser.add_argument(
        "--n_train_anomalies", default=100, type=int,
        help="Number of known anomalies in the training set"
    )
    parser.add_argument(
        "--n_epochs", default=100, type=int,
        help="Number of epochs"
    )
    parser.add_argument(
        "--p_pollution", default=0.0, type=float,
        help="Pollution in the training data"
    )
    parser.add_argument(
        "--eval_n_anomalies", default=True, type=bool,
        help="Compare the performance for different n_anomalies"
    )
    parser.add_argument(
        "--eval_p_pollution", default=True, type=bool,
        help="Compare the performance for different p_pollution"
    )
    parser.add_argument(
        "--is_override", default=False, type=bool,
        help="Override the previous models"
    )
    parser.add_argument(
        "--learning_rate", default=.001, type=float,
        help="Learning rate for adam"
    )
    parser.add_argument(
        "--model_path", default=BASE_PATH / "models", type=Path, help="Base output path for the models"
    )
    parser.add_argument(
        "--result_path", default=None, type=Path,
        help="Base output path for the results, if None use the model path"
        # default = BASE_PATH / "results"
    )

    return parser
