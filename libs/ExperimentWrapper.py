import pickle
import random
import re

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.eager.context import eager_mode

from typing import Union, List, Tuple, Dict, NoReturn
from pathlib import Path
from copy import deepcopy

from libs.GAA import GAA
from libs.DataHandler import DataLabels
from libs.Metrics import evaluate_roc, roc_to_pandas
from libs.architecture.target import Autoencoder
from libs.constants import BASE_PATH

from sklearn.metrics import roc_curve


class ExperimentWrapper:
    def __init__(
            self, data_setup: List[DataLabels],
            random_seed: int = None, is_override=False,
            save_prefix: str = '', out_path: Path = BASE_PATH, auto_subfolder: bool = True
    ):
        """
        Wrapper class to have a common scheme for the experiments
        :param data_setup: data configuration for every experiment
        :param save_prefix: prefix for saved NN models
        :param random_seed: seed to fix the randomness
        :param is_override: override output if it already exists
        :param out_path: output base path for the models, usually the base path
        :param auto_subfolder: create a subfolder on the out_path with the random seed
        """

        # Save the parameter grid
        self.data_setup = data_setup  # This is mutable to allow target splitting

        # Configuration
        self.is_override = is_override

        # Folder paths
        self.out_path = out_path
        if auto_subfolder:
            self.out_path /= f"{random_seed}"
        # If necessary, create the output folder
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_prefix = f"{save_prefix}"
        if random_seed is not None:
            self.save_prefix += f"_{random_seed}"

        # Fix randomness
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        # Alright, we can't make the NN deterministic on a GPU [1]. Probably makes more sense to keep the sample
        # selection deterministic, but repeat all NN-related aspects.
        # [1] https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        # tf.random.set_seed(random_seed)

    def evaluate_ad(
            self, data_split: str,
            architecture_params: dict = None, target_params: dict = None,
            evaluate_baseline: Dict[str, dict] = None, out_path: Path = None
    ) -> NoReturn:
        """
        Evaluate the performance of the anomaly detection method
        :param data_split: data split, e.g. val or test
        :param architecture_params: architecture settings of the anomaly detection method
        :param target_params: target network's architecture settings
        :param evaluate_baseline: also evaluate the given baseline methods, expects a dict of {"baseline": {config}}
        :param out_path: special output path for the results
        :return:
        """

        if architecture_params is None: architecture_params = {}
        if target_params is None: target_params = {}
        if out_path is None: out_path = self.out_path

        for i_data, cur_data in enumerate(deepcopy(self.data_setup)):
            # Start with a new session
            plt.clf()
            keras.backend.clear_session()

            # We'll output the metrics and the x,y coordinates for the ROC
            df_metric = pd.DataFrame(columns=["AUC", "AP"])
            df_roc = pd.DataFrame()

            # Announce what we're doing
            ad_prefix = self.parse_name(cur_data, is_supervised=True, is_eval=True, suffix=f"split:{data_split}")
            print(f"Now evaluating {ad_prefix}")

            # Get the output path
            csv_path = self.get_model_path(
                base_path=out_path, file_name=ad_prefix,
                file_suffix=".csv"
            )

            # Evaluate baseline methods
            if evaluate_baseline:
                for baseline_name, baseline_config in evaluate_baseline.items():
                    keras.backend.clear_session()
                    baseline_metric, baseline_roc = self.evaluate_baseline_on(
                        data_split=data_split, baseline=baseline_name, input_config=cur_data, **baseline_config
                    )
                    df_metric.loc[baseline_name, :] = baseline_metric
                    df_roc = pd.concat([df_roc, baseline_roc], axis=1, ignore_index=False)

            # Save the resulting DFs
            df_metric.to_csv(csv_path.with_suffix(".metric.csv"))
            df_roc.to_csv(csv_path.with_suffix(".roc.csv"))

            # Plot the ROC
            plt.plot([0, 1], [0, 1], label="Random Classifier")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            # plt.show()
            try:
                plt.savefig(csv_path.with_suffix(".roc.png"))
            except RuntimeError:
                # LaTeX used, but not available - this should not interfere with the rest of the evaluation
                pass

    # -- Baselines --
    @staticmethod
    def _get_baseline_info(baseline: str) -> Tuple[str, bool]:
        """
        Get the right file suffix for the respective baseline
        :param baseline: baseline name
        :return: file suffix and if the method is supervised
        """
        if baseline in ["AE", "GAA-AE", "DeepSVDD-AE", "DeepSVDD", "GradCon"]:
            file_suffix = ".tf"
            is_supervised = False
        elif baseline in ["A3", "DevNet", "DeepSAD", "GAA", "GAA-1", "GAA-2", "GAA-3", "GAA-4"]:
            file_suffix = ".tf"
            is_supervised = True
        else:
            raise NotImplementedError(f"{baseline} is not a known baseline method")

        return file_suffix, is_supervised

    def train_baseline(
            self, baseline: str,
            compile_params: dict = None, fit_params: dict = None, **model_params
    ) -> NoReturn:
        """
        Train some baseline methods
        :param baseline: which baseline method to evaluate
        :param compile_params: arguments for the (optional) compile function
        :param fit_params: arguments for the (optional) fit function
        :param model_params: extra arguments for the baseline method constructor
        :return:
        """

        # Check if baseline method exists
        file_suffix, is_supervised = self._get_baseline_info(baseline)

        # Default to empty dictionaries
        if compile_params is None: compile_params = {}
        if fit_params is None: fit_params = {}

        for cur_data in self.data_setup:
            # Unsupervised methods don't know the training anomalies
            this_prefix = self.parse_name(cur_data, is_supervised=is_supervised)
            print(f"Now training baseline method '{baseline}' for {this_prefix}")

            # Check if the respective model exists
            out_path = self.get_model_path(
                base_path=self.out_path, file_name=this_prefix,
                file_suffix=file_suffix, sub_folder=baseline
            )
            if not self.is_override and (
                    out_path.exists()
                    or out_path.with_suffix(".overall.h5").exists()
                    or out_path.with_suffix(".tf.index").exists()
            ):
                print("This baseline method was already trained. Use is_override=True to override it.")
                continue

            # Create the parent folder if not existing
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=False)

            # Open the data
            this_data = cur_data.to_data(for_experts=False)

            # Fit the baseline method
            if baseline in ["GAA", "GAA-1", "GAA-2", "GAA-3", "GAA-4"]:
                # Load the target autoencoder
                ae_prefix = self.parse_name(cur_data, is_supervised=False)
                # Check if the respective model exists
                ae_path = self.get_model_path(
                    base_path=self.out_path, file_name=ae_prefix,
                    file_suffix=file_suffix, sub_folder="GAA-AE"
                )
                this_ae = keras.models.load_model(ae_path)

                # Create GAA - the number afterwards determines the number of targets
                if baseline == "GAA-1":
                    baseline_model = GAA(**model_params, n_target=1)
                elif baseline == "GAA-2":
                    baseline_model = GAA(**model_params, n_target=2)
                elif baseline == "GAA-3":
                    baseline_model = GAA(**model_params, n_target=3)
                elif baseline == "GAA-4":
                    baseline_model = GAA(**model_params, n_target=4)
                else:
                    baseline_model = GAA(**model_params)
                baseline_model.add_target(this_ae, x_train=this_data.train_target_holdout, batch_size=fit_params["batch_size"])
                baseline_model.compile(**compile_params)

                # We can save computation time by precomputing the gradients (needs more RAM, though)
                if "train_on_gradients" in model_params and model_params["train_on_gradients"] is True:
                    # Call once to set the input shapes
                    baseline_model(this_data.val_alarm[0])
                    # Precompute the gradients to save some time
                    grad_train = baseline_model.call_extract_gradients(this_data.train_alarm[0].astype(np.float32))
                else:
                    grad_train = this_data.train_alarm[0]

                # Fit and save
                baseline_model.fit(
                    x=grad_train, y=this_data.train_alarm[1], **fit_params
                )

                baseline_model.save(out_path)

            elif baseline in ["AE", "GAA-AE", "DeepSVDD-AE"]:
                # Use the right architecture
                if baseline in ["AE", "GAA-AE"]:
                    # GAA also uses simple AEs, but trained with less epochs
                    this_net = Autoencoder(**model_params)
                elif baseline == "DeepSVDD-AE":
                    from baselines.deep_svdd.DeepSVDD import DeepSVDDAE
                    this_net = DeepSVDDAE(**model_params)
                else:
                    raise NotImplementedError("Unknown AE architecture")

                if "optimizer" in compile_params:
                    raise NotImplementedError("Please omit the 'optimizer' keyword. So far only Adam is supported. Specify the LR directly.")
                this_net.compile(**compile_params)

                # For the GAA AE, we keep some data oh holdout
                train_data = this_data.train_target_holdout if baseline == "GAA-AE" else this_data.train_target
                this_net.fit(
                    x=train_data[0], y=train_data[1],
                    validation_data=this_data.val_target,
                    **fit_params
                )

                this_net.save(out_path)

            elif baseline in ["DeepSVDD"]:
                from baselines.deep_svdd.DeepSVDD import DeepSVDD

                # Load the DeepSVDD-AE
                this_ae = keras.models.load_model(out_path.parent.parent / "DeepSVDD-AE" / out_path.name)

                # Create the baseline
                if baseline == "DeepSVDD":
                    baseline_model = DeepSVDD(pretrained_ae=this_ae, **model_params)
                else:
                    raise NotImplementedError

                # Call once to initialise the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.calculate_c(this_data.train_target[0])
                baseline_model.compile(**compile_params)
                baseline_model.fit(
                    x=this_data.train_target[0],
                    **fit_params
                )

                # Save the baseline
                baseline_model.save_weights(out_path)

            elif baseline == "DeepSAD":
                from baselines.deep_svdd.DeepSVDD import DeepSAD

                # Load the DeepSVDD-AE
                ae_prefix = self.parse_name(cur_data, is_supervised=False)
                # Check if the respective model exists
                ae_path = self.get_model_path(
                    base_path=self.out_path, file_name=ae_prefix,
                    file_suffix=file_suffix, sub_folder="DeepSVDD-AE"
                )
                this_ae = keras.models.load_model(ae_path)

                # Create the baseline
                baseline_model = DeepSAD(pretrained_ae=this_ae, **model_params)

                # Call once to initialise the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.calculate_c(this_data.train_target[0])
                baseline_model.compile(**compile_params)
                # DeepSAD expects the labels to be either +1 (normal) or -1 (anomalous)
                y_sad = (-2 * this_data.train_alarm[1].astype(np.int8) + 1).reshape((-1, )).astype(np.float32)
                baseline_model.fit(
                    x=this_data.train_alarm[0],
                    y=y_sad,
                    **fit_params
                )

                # Save the baseline
                baseline_model.save_weights(out_path)

            elif baseline in ["DevNet"]:
                from baselines.devnet_v2.devnet_kdd19 import fit_devnet

                # Revert to TF1 for compatibility
                tf.compat.v1.disable_v2_behavior()
                try:
                    baseline_model = fit_devnet(
                        random_state=self.random_seed,
                        x=this_data.train_alarm[0].reshape(this_data.train_alarm[0].shape[0], -1),
                        y=this_data.train_alarm[1].reshape(this_data.train_alarm[1].shape[0], -1),
                    )
                    baseline_model.save_weights(str(out_path))
                except ValueError:
                    print("Error fitting DevNet. Are there any known anomalies available?")

                # We need to restore the TF2 behaviour afterwards
                tf.compat.v1.reset_default_graph()
                tf.compat.v1.enable_v2_behavior()

            elif baseline == "A3":
                from baselines.a3.A3v2 import A3

                # Load the target autoencoder
                ae_prefix = self.parse_name(cur_data, is_supervised=False)
                # Check if the respective model exists
                ae_path = self.get_model_path(
                    base_path=self.out_path, file_name=ae_prefix,
                    file_suffix=file_suffix, sub_folder="AE"
                )
                this_ae = keras.models.load_model(ae_path)

                # Create A3
                baseline_model = A3(m_target=this_ae, **model_params)
                baseline_model.compile(**compile_params)

                # Fit and save
                baseline_model.fit(
                    x=this_data.train_alarm[0], y=this_data.train_alarm[1],
                    validation_data=this_data.val_alarm, **fit_params
                )

                baseline_model.save(out_path)

            elif baseline == "GradCon":
                from baselines.gradcon.GradCon import GradConCAE

                # Due to some if & for statements, we need eager mode:
                with eager_mode():
                    # Use pretrained AE as model, cloning encoder and decoder
                    this_ae = keras.models.load_model(out_path.parent.parent / "AE" / out_path.name)

                    # Create model
                    baseline_model = GradConCAE(pretrained_ae=this_ae)
                    baseline_model.compile(**compile_params)

                    # Fit
                    baseline_model.fit(
                        x=this_data.train_target[0], y=this_data.train_target[1],
                        validation_data=this_data.val_target, **fit_params
                    )

                    # Save model and averages
                    baseline_model.save_weights(out_path)

    def evaluate_baseline_on(
            self, data_split: str, baseline: str, input_config, **model_params
    ) -> Tuple[list, pd.DataFrame]:
        """
        Evaluate a baseline method on a given data config
        :param data_split: data split, e.g. val or test
        :param baseline: which baseline method to evaluate
        :param input_config: configuration the baseline is evaluated on (takes test data)
        :param model_params: extra arguments for the baseline method constructor
        :return: DataFrame containing the metrics & DataFrame containing the ROC x,y data
        """

        # Check if baseline method exists
        file_suffix, is_supervised = self._get_baseline_info(baseline)

        this_prefix = self.parse_name(input_config, is_supervised=is_supervised)

        # Handle the file origins
        in_path = self.get_model_path(
            base_path=self.out_path, file_name=this_prefix,
            file_suffix=file_suffix, sub_folder=baseline
        )

        # Open the baseline and predict
        this_data = input_config.to_data(test_type=data_split, for_experts=False)
        pred_y = None
        try:
            if baseline in ["GAA", "GAA-1", "GAA-2", "GAA-3", "GAA-4"]:
                # Load GAA
                baseline_model = keras.models.load_model(in_path)

                pred_y = baseline_model.predict(x=this_data.test_alarm[0])

            elif baseline in ["AE"]:

                baseline_model = keras.models.load_model(in_path)

                pred_y = baseline_model.m_dec.predict(
                    baseline_model.m_enc(this_data.test_alarm[0])
                )

                # We'll return the MSE as score
                pred_y = np.square(pred_y - this_data.test_alarm[0])
                # We might have 2D inputs: collapse to one dimension
                pred_y = np.reshape(pred_y, (pred_y.shape[0], -1))
                pred_y = np.mean(pred_y, axis=1)

            elif baseline in ["DeepSAD", "DeepSVDD"]:
                from baselines.deep_svdd.DeepSVDD import DeepSVDD, DeepSAD

                # Load the DeepSVDD-AE
                ae_prefix = self.parse_name(input_config, is_supervised=False)
                # Check if the respective model exists
                ae_path = self.get_model_path(
                    base_path=self.out_path, file_name=ae_prefix,
                    file_suffix=file_suffix, sub_folder="DeepSVDD-AE"
                )
                this_ae = keras.models.load_model(ae_path)

                # Create the baseline
                if baseline == "DeepSAD":
                    baseline_model = DeepSAD(pretrained_ae=this_ae, **model_params)
                elif baseline == "DeepSVDD":
                    baseline_model = DeepSVDD(pretrained_ae=this_ae, **model_params)
                else:
                    raise NotImplementedError

                # Call once to initialise the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.load_weights(in_path)
                # c isn't saved, thus we need to recalculate is based on the training data
                baseline_model.calculate_c(this_data.train_target[0])

                # Predict
                pred_y = baseline_model.score(this_data.test_alarm[0])

            elif baseline in ["DevNet"]:
                from baselines.devnet_v2.devnet_kdd19 import predict_devnet

                pred_y = predict_devnet(
                    model_name=str(in_path),
                    x=this_data.test_alarm[0].reshape(this_data.test_alarm[0].shape[0], -1)
                )

            elif baseline == "A3":
                from baselines.a3.A3v2 import A3

                # Load A3
                baseline_model = keras.models.load_model(in_path)

                pred_y = baseline_model.predict(x=this_data.test_alarm[0])

            elif baseline == "GradCon":
                from baselines.gradcon.GradCon import GradConCAE
                # Load model
                this_ae = keras.models.load_model(in_path.parent.parent / "AE" / in_path.name)
                baseline_model = GradConCAE(pretrained_ae=this_ae, **model_params)

                baseline_model(this_data.val_target[0])
                baseline_model.load_weights(in_path)

                pred_y = baseline_model.compute_anomaly_score(this_data.test_alarm[0].astype("float32"))

            else:
                raise NotImplementedError("Unknown baseline method")

        except FileNotFoundError:
            print(f"No model for {baseline} found. Aborting.")
            return [None, None], pd.DataFrame()

        # Plot ROC
        fpr, tpr, thresholds = roc_curve(
            y_true=this_data.test_alarm[1], y_score=pred_y
        )
        plt.plot(fpr, tpr, label=baseline)

        # Generate the output DF
        df_metric = evaluate_roc(pred_scores=pred_y, test_alarm=this_data.test_alarm)
        df_roc = roc_to_pandas(fpr=fpr, tpr=tpr, suffix=baseline)

        return df_metric, df_roc

    def do_everything(
            self, dim_target: Union[List[int], None], dim_alarm: List[int],
            learning_rate: float, batch_size: int, n_epochs: int,
            out_path: Path, evaluation_split: str = "val",
            gaa_precompute_gradient: bool = False,
            gaa_versions: List[str] = ("GAA-1", "GAA-2", "GAA-3", "GAA-4"),
            gaa_learning_rate: float = None, gaa_n_epochs: int = None, gaa_ae_n_epochs: int = 10,
    ) -> NoReturn:
        """
        Train & evaluate the anomaly detection method and all relevant baseline methods
        :param dim_target: dimensions of the autoencoder's encoder (decoder is symmetric to this)
        :param dim_alarm: dimensions of the alarm network
        :param learning_rate: training learning rate for Adam
        :param batch_size: training batch size
        :param n_epochs: training epochs
        :param out_path: output path for the evaluation
        :param evaluation_split: data split to evaluate the methods on
        :param gaa_names: versions of GAA used for the ablation study
        :param gaa_precompute_gradient: convert the training data to gradients (speeds up computation, but needs more RAM)
        :param gaa_learning_rate: learning rate just for GAA, use the same as for everything else if None
        :param gaa_n_epochs: number of epochs just for GAA, use the same as for everything else if None
        :param gaa_ae_n_epochs: number of epochs just for the GAA AE, use the same as for everything else if None
        :return:
        """

        self.train_baseline(
            baseline="AE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )

        # Either use the given training values or the standard ones
        gaa_ae_n_epochs = gaa_ae_n_epochs if gaa_ae_n_epochs is not None else n_epochs
        gaa_n_epochs = gaa_n_epochs if gaa_n_epochs is not None else n_epochs
        gaa_learning_rate = gaa_learning_rate if gaa_learning_rate is not None else learning_rate

        # Train GAA
        self.train_baseline(
            baseline="GAA-AE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": gaa_ae_n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        # Train multiple versions of GAA, each with different number of targets
        for cur_gaa in gaa_versions:
            self.train_baseline(
                baseline=cur_gaa,
                layer_dims=dim_alarm, train_on_gradients=gaa_precompute_gradient,
                compile_params={"learning_rate": gaa_learning_rate},
                fit_params={"epochs": gaa_n_epochs, "batch_size": batch_size, "verbose": 2},
            )

        # Baseline methods
        self.train_baseline(
            baseline="A3",
            layer_dims=dim_alarm,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2}
        )
        self.train_baseline(
            baseline="DeepSVDD-AE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="DeepSAD",
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="DevNet",
            fit_params={"epochs": n_epochs, "batch_size": batch_size},
        )

        self.train_baseline(
            baseline="GradCon",
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )

        # Get the results
        baseline_methods = {
            "AE": {},
            "A3": {},
            "DeepSAD": {},
            "DevNet": {},
            "GradCon": {},
        }
        # Add all GAAs
        for cur_gaa in gaa_versions:
            baseline_methods[cur_gaa] = {"train_on_gradients": gaa_precompute_gradient}

        self.evaluate_ad(
            evaluation_split, out_path=out_path, architecture_params={"layer_dims": dim_alarm},
            evaluate_baseline=baseline_methods
        )

    # -- Helpers --
    @staticmethod
    def parse_name(
            in_conf: DataLabels, prefix: str = None, suffix: str = None,
            is_supervised: bool = False, is_eval: bool = False
    ) -> str:
        """
        Convert configuration to a nicer file name
        :param in_conf: dictionary
        :param prefix: a string that will be prepended to the name
        :param suffix: a string that will be appended to the name
        :param is_supervised: is the method supervised? if so also save the number of training samples
        :param is_eval: requesting a name for the metric files if so also save the anomalies we're evaluating on
        :return: string describing the dictionary
        """
        # Based on the method, we need more parameters in the file name
        keep_keywords = ["y_norm", "p_pollution"]
        if is_supervised:
            keep_keywords += ["y_anom_train", "n_train_anomalies"]
        if is_eval:
            keep_keywords += ["y_anom_test"]

        # Convert to member dict if it's not a dict
        out_dict = in_conf if isinstance(in_conf, dict) else vars(in_conf).copy()

        # Remove all keywords but the desired ones
        out_dict = {
            cur_key: cur_val for cur_key, cur_val in out_dict.items() if cur_key in keep_keywords
        }

        # Parse as string
        out_str = str(out_dict)

        # Remove full stops and others as otherwise the path may be invalid
        out_str = re.sub(r"[{}\\'.<>\[\]()\s]", "", out_str)

        # Alter the string
        if prefix: out_str = prefix + "_" + out_str
        if suffix: out_str = out_str + "_" + suffix

        return out_str

    @staticmethod
    def dict_to_str(in_dict: dict) -> str:
        """
        Parse the values of a dictionary as string
        :param in_dict: dictionary
        :return: dictionary with the same keys but the values as string
        """
        out_dict = {cur_key: str(cur_val) for cur_key, cur_val in in_dict.items()}

        return out_dict

    def get_model_path(
            self, base_path: Path,
            file_name: str = None, file_suffix: str = ".tf",
            sub_folder: str = "", sub_sub_folder: str = "",
    ) -> Path:
        """
        Get the path to save the NN models
        :param base_path: path to the project
        :param file_name: name of the model file (prefix is prepended)
        :param file_suffix: suffix of the file
        :param sub_folder: folder below model folder, e.g. for alarm/target
        :param sub_sub_folder: folder below subfolder, e.g. architecture details
        :return:
        """
        out_path = base_path

        if sub_folder:
            out_path /= sub_folder

        if sub_sub_folder:
            out_path /= sub_sub_folder

        # Create the path if it does not exist
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=False)

        if file_name:
            out_path /= f"{self.save_prefix}_{file_name}"
            out_path = out_path.with_suffix(file_suffix)

        return out_path

