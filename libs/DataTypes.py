from dataclasses import dataclass

@dataclass
class ExperimentData:
    """
    All data needed for the experiment
    """
    train_target: dict
    train_target_holdout: dict
    train_alarm: tuple
    val_target: dict
    val_alarm: tuple
    test_target: dict
    test_alarm: tuple
    data_shape: tuple
    input_shape: tuple


