import tensorflow as tf
from argparse import ArgumentParser

from libs.DataHandler import URL
from libs.ExperimentWrapper import ExperimentWrapper

from libs.constants import add_standard_arguments, ALARM_SMALL, ALARM_BIG, N_TRAIN_ANOMALIES, P_POLLUTIONS

# Reduce the hunger of TF when we're training on a GPU
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

# Configuration
this_parse = ArgumentParser(description="Train R2-AD2 on URL")
add_standard_arguments(this_parse)
this_args = this_parse.parse_args()

experiment_config = [
    URL(
        random_state=this_args.random_seed, y_normal=["benign"],
        y_anomalous=["Defacement", "malware", "phishing", "spam"],
        n_train_anomalies=this_args.n_train_anomalies, p_pollution=this_args.p_pollution
    ),
    URL(
        random_state=this_args.random_seed, y_normal=["benign"],
        y_anomalous=["Defacement", "malware", "phishing", "spam"],
        y_anomalous_train=["Defacement", "malware"],
        n_train_anomalies=this_args.n_train_anomalies, p_pollution=this_args.p_pollution
    ),
]

if this_args.eval_n_anomalies:
    for cur_n_anomalies in N_TRAIN_ANOMALIES:
        experiment_config.append(
            URL(
                random_state=this_args.random_seed, y_normal=["benign"],
                y_anomalous=["Defacement", "malware", "phishing", "spam"],
                n_train_anomalies=cur_n_anomalies, p_pollution=this_args.p_pollution
            )
        )
if this_args.eval_p_pollution:
    for cur_p_pollution in P_POLLUTIONS:
        experiment_config.append(
            URL(
                random_state=this_args.random_seed, y_normal=["benign"],
                y_anomalous=["Defacement", "malware", "phishing", "spam"],
                n_train_anomalies=this_args.n_train_anomalies, p_pollution=cur_p_pollution
            )
        )

DIM_TARGET = (60, 30, 15)
DIM_ALARM = ALARM_BIG
BATCH_SIZE = 64

if __name__ == '__main__':

    this_experiment = ExperimentWrapper(
        save_prefix="URL", data_setup=experiment_config,
        random_seed=this_args.random_seed, out_path=this_args.model_path,
    )

    this_experiment.do_everything(
        dim_target=DIM_TARGET, dim_alarm=DIM_ALARM,
        learning_rate=this_args.learning_rate, batch_size=BATCH_SIZE, n_epochs=this_args.n_epochs,
        out_path=this_args.result_path, evaluation_split=this_args.data_split
    )