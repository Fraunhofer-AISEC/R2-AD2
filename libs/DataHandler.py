import abc

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff, loadmat

from typing import List, Callable, Union, Tuple, Dict, NoReturn

from libs.DataTypes import ExperimentData
from libs.constants import BASE_PATH


class DataLabels:
    """
    Class storing test/train data
    """

    def __init__(
            self, x_train: np.ndarray, y_train: np.ndarray,
            y_normal: Union[List[int], List[str]], y_anomalous: Union[List[int], List[str]],
            y_anomalous_train: Union[List[int], List[str]] = None,
            x_test: np.ndarray = None, y_test: np.ndarray = None,
            x_val: np.ndarray = None, y_val: np.ndarray = None,
            p_test: float = .2, p_val: float = .05, p_pollution: float = 0.0, n_train_anomalies: int = None,
            random_state: int = None
    ):

        # We'll put everything in the train data if no test data was given and split later
        self.x_train: np.ndarray = x_train  # Train data
        self.y_train: np.ndarray = y_train
        self.x_test: np.ndarray = x_test  # Test data
        self.y_test: np.ndarray = y_test
        self.x_val: np.ndarray = x_val  # Validation data
        self.y_val: np.ndarray = y_val

        # Store normal and anomalous data globally
        self.y_norm: Union[List[int], List[str]] = y_normal
        self.y_anom_test: Union[List[int], List[str]] = y_anomalous
        # If train anomalies were not specified, use the ones for testing
        self.y_anom_train: Union[List[int], List[str]] = y_anomalous_train if y_anomalous_train is not None else y_anomalous
        self.y_anom_all: Union[List[int], List[str]] = list(set(self.y_anom_test) | set(self.y_anom_train))
        self.y_all = list(set(self.y_norm) | set(self.y_anom_all))
        # Sanity check: we should not have overlapping classes
        assert len(self.y_all) == len(y_normal + y_anomalous), "Normal and anomalous classes must not overlap"

        # If needed: a scaler
        self.scaler: TransformerMixin = None

        # Configuration
        self.test_split: float = p_test  # Test data percentage
        self.val_split: float = p_val  # Validation data percentage
        self.p_pollution: float = p_pollution
        self.n_train_anomalies: int = n_train_anomalies
        self.random_state = random_state
        self.random_gen = np.random.default_rng(random_state)

        # Metadata
        self.shape: tuple = None  # Shape of the data

        # Fill the values
        self._post_init()

    ## Class methods
    def __repr__(self):
        return self.__class__.__name__

    ## Retrievers
    def get_target_autoencoder_data(
            self, data_split: str, p_holdout: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data useful for autoencoders
        :param data_split: get data of either "train", "val" or "test"
        :param p_holdout: percentage of data that is not used during training, i.e. kept "fresh" for the alarm network
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split, only_normal=True)
        this_x = this_data[0]

        # Keep some fresh data for the alarm network
        if p_holdout:
            n_samples = this_x.shape[0]
            n_keep = int((1 - p_holdout) * n_samples)
            idx_keep = self.random_gen.choice(n_samples, n_keep, replace=False)
            this_x = this_x[idx_keep, :]

        # For the autoencoder, we don't need much else than x
        return this_x, this_x

    def get_target_classifier_data(
            self, data_split: str, only_normal: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for useful for classifiers
        :param data_split: get data of either "train", "val" or "test"
        :param only_normal: get labels of normal data points only
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split, only_normal=only_normal)
        this_x = this_data[0]
        this_y = this_data[1]

        # Return the data
        return this_x, this_y

    def get_alarm_data(
            self, data_split: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the labels for the alarm network, i.e. with binary anomaly labels
        :param data_split: get data of either "train", "val" or "test"
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split, only_normal=False)
        this_x = this_data[0]

        # Make labels binary
        # (train & test anomalies were filtered earlier, thus we're left with only the relevant samples)
        this_y = np.isin(this_data[1], self.y_anom_all)
        this_y = this_y.astype("uint8")

        return this_x, this_y

    @staticmethod
    def _idx_anom_samples(y: np.ndarray, n_anomaly_samples: int = None) -> np.ndarray:
        """
        Give the indices that should be deleted due to less anomaly samples
        :param y: binary array
        :param n_anomaly_samples: amount of anomaly samples that should be kept
        :return: indices about to be deleted
        """
        # Don't delete anything if None is given
        if n_anomaly_samples is None:
            return np.array([])

        # IDs of all anomaly samples
        idx_anom = np.where(y == 1)[0]

        # Select the indices to delete
        n_delete = len(idx_anom) - n_anomaly_samples
        # assert n_delete > 0  # Not true in the unsupervised case where we don't use any
        idx_delete = np.random.choice(idx_anom, size=n_delete, replace=False)

        return idx_delete

    def get_attention_labels(
            self, data_split: str, add_global: bool = True
    ) -> np.ndarray:
        """
        Get the labels for the attention network, i.e. the position of the target networks
        :param data_split: get data of either "train", "val" or "test"
        :param add_global: add an extra column for the anomaly expert
        :return: labels
        """

        # Note down all available labels
        all_labels = self.y_norm.copy()
        anomaly_label = "anom" if isinstance(all_labels[0], str) else -1
        assert anomaly_label not in all_labels, "The used anomaly label may overwrite existing ones"
        if add_global:
            all_labels.append(anomaly_label)

        # We transform the classification labels to a 1-Hot matrix
        y_cat = self.get_target_classifier_data(data_split=data_split, only_normal=False)[1]
        # Important: must be signed int, otherwise -1 = 255 or whatever the range is
        if anomaly_label == -1:
            y_cat = y_cat.astype(np.int16)
        # Summarise all known anomalies to one "other" class
        y_cat[np.where(np.isin(y_cat, self.y_anom_test))] = anomaly_label
        # Use Pandas for easier mapping
        y_cat = pd.Series(y_cat[:, 0], dtype=pd.api.types.CategoricalDtype(categories=all_labels, ordered=False))
        this_y = pd.get_dummies(y_cat)

        return this_y.to_numpy()

    def get_mae_data(
            self, data_split: str, equal_size: bool = True
    ) -> Dict[Union[int, str], tuple]:
        """
        Get multi-autoencoder data, i.e. AE data distributed among experts
        :param data_split: get data of either "train", "val" or "test"
        :param equal_size: equalise the amount of samples for each expert by oversampling, i.e. we repeat samples in the smaller sets
        :return: features and labels for each data class
        """

        # We basically loop through all normal labels
        this_x, this_y = self.get_target_classifier_data(data_split=data_split, only_normal=True)

        expert_data = {}
        for cur_label in self.y_norm:
            idx_label = np.where(this_y == cur_label)[0]
            expert_data[cur_label] = (this_x[idx_label, :], this_x[idx_label, :])

        # If we want to equalise the amount of samples, we'll repeat some of them
        if equal_size:
            expert_data = self.equalise_expert_data(expert_data)

        return expert_data

    ## Preprocessors
    def _post_init(self):
        """
        Process the data
        :return:
        """

        # Check if all classes are covered
        available_classes = np.unique(self.y_train).tolist()
        assert set(available_classes) == set(self.y_all), \
            "There are classes in training data that are not covered by the normal&anomalous samples. Is this intended?"

        # The labels should not have an empty dimension
        if len(self.y_train.shape) == 1:
            self.y_train = np.expand_dims(self.y_train, axis=-1)
        if (self.y_val is not None) and (len(self.y_val.shape) == 1):
            self.y_val = np.expand_dims(self.y_val, axis=-1)
        if (self.y_test is not None) and (len(self.y_test.shape) == 1):
            self.y_test = np.expand_dims(self.y_test, axis=-1)

        # Split in test and train
        # Fix the test set to one seed to simulate the "split and forget" nature of it
        if self.x_test is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train, self.y_train, test_size=self.test_split, random_state=42
            )

        # Split in train and validation
        if self.x_val is None:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_train, self.y_train, test_size=self.val_split, random_state=self.random_state
            )

        # Pollute the data
        if self.p_pollution: self._pollute_data()
        # Reduce amount of training anomalies and filter for the given classes
        self._reduce_train_anomalies()
        # Filter test anomalies for the given classes
        self._reduce_test_anomalies()
        # Preprocess
        self._preprocess()
        # Note down the shape
        self.shape = self.x_train.shape[1:]

    @abc.abstractmethod
    def _preprocess(self):
        # Preprocessing steps, e.g. data normalisation
        raise NotImplementedError("Implement in subclass")

    def _pollute_data(self) -> NoReturn:
        """
        Change the label of some anomalous training samples to random normal training labels
        :return:
        """
        # Filter for normal an anomalous samples
        idx_normal = np.where(np.isin(self.y_train, self.y_norm))[0]
        # We use use all anomalous sample types for the pollution
        idx_anomalous = np.where(np.isin(self.y_train, self.y_anom_all))[0]

        # Calculate how many samples we need for the pollution
        n_pollute = self._contaminate_p_to_n(self.p_pollution, idx_normal.shape[0])
        # Check if we have enough samples
        need_duplicates = n_pollute > idx_anomalous.shape[0]
        if need_duplicates:
            print(f"For the desired pollution level, {n_pollute} samples are required, "
                  f"but there are only {len(idx_anomalous)} known anomalies. There will be duplicates.")

        # Pollution => Assign random normal labels to some anomalous samples
        idx_pollute = self.random_gen.choice(idx_anomalous, n_pollute, replace=need_duplicates)
        self.y_train[idx_pollute, :] = self.random_gen.choice(self.y_norm, self.y_train[idx_pollute].shape, replace=True)

    def _reduce_train_anomalies(self, raise_error: bool = True) -> NoReturn:
        """
        Use less anomaly samples during training
        :param raise_error: raise an error if not enough known anomalies are available
        :return:
        """
        # Filter for normal an anomalous samples
        idx_norm = np.where(np.isin(self.y_train, self.y_norm))[0]
        idx_anom_train = np.where(np.isin(self.y_train, self.y_anom_train))[0]

        # Check if enough anomalies are available
        need_duplicates = self.n_train_anomalies > idx_anom_train.shape[0]
        if need_duplicates:
            error_text = f"Expected {self.n_train_anomalies} training anomalies, but only {idx_anom_train.shape[0]} are available."
            if raise_error:
                assert not need_duplicates, error_text
            else:
                print(error_text + " There will be duplicates.")

        # Reduce the number of anomalies
        if self.n_train_anomalies is not None:
            idx_anom_train = self.random_gen.choice(idx_anom_train, self.n_train_anomalies, replace=need_duplicates)
        # Concatenate with normal samples to get all samples to keep
        idx_keep = np.concatenate([idx_norm, idx_anom_train], axis=0)

        # Filter
        self.x_train = self.x_train[idx_keep, :]
        self.y_train = self.y_train[idx_keep, :]

    def _reduce_test_anomalies(self) -> NoReturn:
        """
        Filter to the desired train/test anomalies
        :return:
        """
        # Filter val for the train anomalies (we don't know about the unknown anomalies)
        idx_anom_val = np.where(np.isin(self.y_val, self.y_norm + self.y_anom_train))[0]
        # Filter test for the test anomalies (to evaluate if our method transfers to unknown anomalies)
        idx_anom_test = np.where(np.isin(self.y_test, self.y_norm + self.y_anom_test))[0]

        # Filter
        self.x_val = self.x_val[idx_anom_val, :]
        self.y_val = self.y_val[idx_anom_val, :]
        self.x_test = self.x_test[idx_anom_test, :]
        self.y_test = self.y_test[idx_anom_test, :]

    ## Helpers
    def include_to_drop(self, include_data: Union[List[int], List[str]]) -> Union[List[int], List[str]]:
        """
        Convert a list of classes to include to a list of classes to drop
        :param include_data: classes to include
        :param all_classes: available classes
        :return: classes to drop
        """

        drop_classes = set(self.available_classes) - set(include_data)

        return list(drop_classes)

    @staticmethod
    def equalise_expert_data(expert_data: dict) -> dict:
        """
        Equalise the length of all expert clusters by repeating the entries
        :param expert_data: dictionary indexed by the expert clusters
        :return: dictionary indexed by the expert clusters, but equally sized data
        """
        max_size = max([cur_val[0].shape[0] for cur_val in expert_data.values()])

        for cur_idx, cur_val in expert_data.items():
            # Nothing to equalise if it's the same size
            if cur_val[0].shape[0] == max_size:
                continue

            # Sample random items and add the to the data
            cur_size = cur_val[0].shape[0]
            idx_repeat = np.random.choice(cur_size, size=max_size - cur_size)

            extended_x = np.concatenate(
                [cur_val[0], cur_val[0][idx_repeat, :]], axis=0
            )
            extended_y = np.concatenate(
                [cur_val[1], cur_val[1][idx_repeat, :]], axis=0
            )

            expert_data[cur_idx] = (extended_x, extended_y)

        return expert_data

    def to_data(
            self, train_type: str = "train", test_type: str = "test",
            for_experts: bool = False, equal_size: bool = False, p_holdout: float = .25
    ) -> ExperimentData:
        """
        Convert the configuration to actual data
        :param train_type: use the train or validation data for training (only used to load less data while debugging)
        :param test_type: use the test or validation data for evaluation (i.e. code once, use twice)
        :param for_experts: split the target data among classes
        :param equal_size: equalise the amount of samples for each expert by oversampling, i.e. we repeat samples in the smaller sets
        :param p_holdout: percentage of data that is not used during training, i.e. kept "fresh" for the alarm network
        """

        this_data = ExperimentData(
            # Target training: all normal samples
            train_target=self.get_mae_data(
                data_split=train_type, equal_size=equal_size,
            ) if for_experts else self.get_target_autoencoder_data(
                data_split=train_type
            ),
            train_target_holdout=self.get_target_autoencoder_data(
                data_split=train_type, p_holdout=p_holdout
            ),
            train_alarm=self.get_alarm_data(
                data_split=train_type
            ),
            val_target=self.get_mae_data(
                data_split="val", equal_size=equal_size,
            ) if for_experts else self.get_target_autoencoder_data(
                data_split="val"
            ),
            val_alarm=self.get_alarm_data(
                data_split="val"
            ),
            # Target testing: all normal samples plus the test anomalies
            test_target=self.get_mae_data(
                data_split=test_type, equal_size=equal_size,
            ) if for_experts else self.get_target_autoencoder_data(
                data_split=test_type
            ),
            test_alarm=self.get_alarm_data(
                data_split=test_type
            ),
            # Shape to generate networks
            data_shape=self.shape,
            input_shape=(None, ) + self.shape
        )

        return this_data

    @staticmethod
    def _contaminate_p_to_n(p_contaminate: float, n_samples: int) -> int:
        """
        Determine the amount of samples needed to be added such that they make up p_train_contamination% of the resulting set
        :param p_contaminate: fraction of contaminated samples in the resulting data set
        :param n_samples: number of samples so far
        :return: number of contaminated samples that need to be added
        """

        assert 0 < p_contaminate < 1

        # Oh dear, percentages: we need to add a higher fraction of samples to get the desired fraction in the new set
        p_add = p_contaminate / (1-p_contaminate)
        n_contaminate = round(n_samples * p_add)

        return n_contaminate

    def _get_data_set(self, data_split: str, only_normal: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the right data split
        :param data_split: train, val or test data?
        :param only_normal: return normal data only (note that normal also includes unknown anomalies when polluted)
        :return: the desired data set
        """

        if data_split == "train":
            this_data = (self.x_train.copy(), self.y_train.copy())

        elif data_split == "test":
            this_data = (self.x_test.copy(), self.y_test.copy())

        elif data_split == "val":
            this_data = (self.x_val.copy(), self.y_val.copy())

        else:
            raise ValueError("The requested data must be of either train, val or test set.")

        # Maybe filter for normal data
        if only_normal:
            idx_normal = np.where(np.isin(this_data[1], self.y_norm))[0]
            this_data = (this_data[0][idx_normal, :], this_data[1][idx_normal, :])

        return this_data

    def scikit_scale(self, scikit_scaler: Callable[[], TransformerMixin] = MinMaxScaler, plus_min_one: bool = False):
        """
        Apply a scikit scaler to the data, e.g. MinMaxScaler transform data to [0,1]
        :return:
        """
        # Fit scaler to train set
        self.scaler = scikit_scaler(feature_range=(-1, 1)) if plus_min_one is True else scikit_scaler()
        # Only use the normal data as otherwise the scaling would be dependent on the number of anomalies
        self.scaler.fit(self.x_train[(np.isin(self.y_train, self.y_norm)).flatten(), :])

        # Scale the rest
        self.x_train = self.scaler.transform(self.x_train)
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)

        pass


class MNIST(DataLabels):
    def __init__(
            self, special_name: str = None,
            *args, **kwargs
    ):
        """
        Load the (fashion) MNIST data set
        :param special_name: load an MNIST-like data set instead (e.g. fashion), use good old MNIST if None
        """
        assert special_name in ["fashion"] or special_name is None

        # Simply load the data with the kind help of Keras
        if special_name == "fashion":
            this_data = tf.keras.datasets.fashion_mnist.load_data()
        else:
            this_data = tf.keras.datasets.mnist.load_data()
        # Extract the data parts
        (x_train, y_train), (x_test, y_test) = this_data

        # Add channel dimension to the data
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        super(MNIST, self).__init__(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, *args, **kwargs
        )

    def _preprocess(self):
        """
        For MNIST, we can scale everything by just dividing by 255
        :return:
        """
        self.x_train = self.x_train / 255.
        self.x_test = self.x_test / 255.
        self.x_val = self.x_val / 255.


class CreditCard(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "creditcard" / "creditcard").with_suffix(".csv"),
            *args, **kwargs
    ):
        """
        Load the CreditCard data set (https://www.kaggle.com/mlg-ulb/creditcardfraud)
        :param data_path: absolute path to the CreditCard csv
        """

        data = pd.read_csv(data_path)

        # Time axis does not directly add information (although frequency might be a feature)
        data = data.drop(['Time'], axis=1)

        # Column class has the anomaly values, the rest is data
        y_train = data.pop("Class")

        # We don't need the overhead of pandas here
        x_train = data.to_numpy()
        y_train = y_train.to_numpy()

        # Bring y in the right shape
        y_train = np.reshape(y_train, (-1, ))

        super(CreditCard, self).__init__(
            x_train=x_train, y_train=y_train, *args, **kwargs
        )

    def _preprocess(self):
        """
        Standardscale the data
        :return:
        """
        self.y_test = self.y_test.astype(np.int)
        self.y_train = self.y_train.astype(np.int)
        self.y_val = self.y_val.astype(np.int)
        self.scikit_scale()


class CovType(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "covtype" / "covtype").with_suffix(".csv"),
            *args, **kwargs
    ):
        """
        Load the covertype data set (https://archive.ics.uci.edu/ml/datasets/Covertype)
        :param data_path: absolute path to the covtype csv
        """

        # Open raw data
        train_data = pd.read_csv(data_path)

        x_train = train_data.drop(columns="Cover_Type")
        y_train = train_data.loc[:, "Cover_Type"]

        super(CovType, self).__init__(
            x_train=x_train.to_numpy(), y_train=y_train.to_numpy(), *args, **kwargs
        )

    def _preprocess(self):
        """
        Standardscale the data
        :return:
        """

        self.scikit_scale()


class Mammography(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "mammography" / "mammography").with_suffix(".mat"),
            *args, **kwargs
    ):
        """
        Load the mammography data set (http://odds.cs.stonybrook.edu/mammography-dataset/)
        :param data_path: absolute path to the mammography.mat
        """

        # Open raw data
        train_data = loadmat(data_path)
        # Extract data
        x = train_data["X"]
        # We take our labels to be scalar
        y = train_data["y"]
        y = np.reshape(y, (-1, ))

        super(Mammography, self).__init__(
            x_train=x, y_train=y, *args, **kwargs
        )

    def _preprocess(self):
        """
        Standardscale the data
        :return:
        """

        self.scikit_scale()


class URL(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "url" / "All").with_suffix(".csv"),
            *args, **kwargs
    ):
        """
        Load the URL 2016 data set (https://www.unb.ca/cic/datasets/url-2016.html)
        :param data_path: absolute path to the All.csv
        """

        # Open raw data
        train_data = pd.read_csv(data_path)

        # There are some NaN entries: we'll drop the columns to be more data-efficient
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(axis=1)

        # The labels are given by the last column
        y_train = train_data.pop("URL_Type_obf_Type")

        super(URL, self).__init__(
            x_train=train_data.to_numpy(), y_train=y_train.to_numpy(),
            *args, **kwargs
        )

    def _preprocess(self):

        self.scikit_scale()


class Darknet(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "darknet" / "Darknet").with_suffix(".CSV"),
            *args, **kwargs
    ):
        """
        Load the Darknet 2020 data set (https://www.unb.ca/cic/datasets/darknet2020.html)
        :param data_path: absolute path to the Darknet.csv
        """

        # Open raw data
        train_data = pd.read_csv(data_path)

        # There are some NaN entries: we'll drop the rows to be more data-efficient
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(axis=0)

        # There are some non-informative features, which may cause overfitting - drop these
        train_data = train_data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'])

        # The labels are given by the labels column
        y_train = train_data.pop("Label")
        # There are some fine graded labels
        y_train_fine = train_data.pop("Label.1")

        super(Darknet, self).__init__(
            x_train=train_data.to_numpy(), y_train=y_train.to_numpy(),
            *args, **kwargs
        )

    def _preprocess(self):
        self.scikit_scale()


class DoH(DataLabels):
    def __init__(
            self, data_path: Path = BASE_PATH / "data" / "doh",
            *args, **kwargs
    ):
        """
        Load the DoH data set (https://www.unb.ca/cic/datasets/dohbrw-2020.html)
        :param data_path: path to the raw .csv files
        """

        # Open raw data
        train_data = [
            pd.read_csv((data_path / "l2-benign").with_suffix(".csv"), index_col="TimeStamp", parse_dates=True),
            pd.read_csv((data_path / "l2-malicious").with_suffix(".csv"), index_col="TimeStamp", parse_dates=True),
        ]
        train_data = pd.concat(train_data)

        # Drop non-informative labels like the IP
        train_data = train_data.drop(columns=[
            "SourceIP", "DestinationIP", "SourcePort", "DestinationPort"
        ])

        # There are some NaN entries: we'll drop the rows to be more data-efficient
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(axis=0)

        # The labels are given by the last column
        y_train = train_data.pop("Label")

        super(DoH, self).__init__(
            x_train=train_data.to_numpy(), y_train=y_train.to_numpy(),
            *args, **kwargs
        )

    def _preprocess(self):

        self.scikit_scale(plus_min_one=False)


class NSL_KDD(DataLabels):
    def __init__(self, data_folder: str = "NSL-KDD", *args, **kwargs):
        """
        NSL KDD data set: https://www.unb.ca/cic/datasets/nsl.html
        :param data_folder: subfolder of "data" where raw data resides
        """

        common_path = BASE_PATH / "data" / data_folder

        # 1) Extract column names: for some reason, the column names are only available in the arff-file
        # This might throw an error as it seems like there are some errors in the .arff file
        # Check that all attributes are either in the format {'a', 'b', 'c'} -> one space after comma
        # or in the format {'a','b','c'} -> no spaces. A mixed format, e.g. {'a','b', 'c'} might not be accepted
        test_data_arff = arff.loadarff((common_path / "KDDTest+").with_suffix(".arff"))
        all_cols = [cur_key for cur_key in test_data_arff[1]._attributes.keys()]
        # The arff also contains the values - we do not need to convert binary flags, though
        all_cat = {
            cur_key: cur_val.range for cur_key, cur_val in test_data_arff[1]._attributes.items()
            if cur_key in ["protocol_type", "service", "flag"]
        }

        # 2) Extract class names: for some reason, the class names are only available in the csv files
        train_data = pd.read_csv((common_path / "KDDTrain+").with_suffix(".txt"), names=all_cols, index_col=False)
        test_data = pd.read_csv((common_path / "KDDTest+").with_suffix(".txt"), names=all_cols, index_col=False)

        # Mark respective columns as categorical
        for cur_key, cur_val in all_cat.items():
            test_data[cur_key] = pd.Categorical(
                test_data[cur_key], categories=cur_val, ordered=False
            )
            train_data[cur_key] = pd.Categorical(
                train_data[cur_key], categories=cur_val, ordered=False
            )

        # Drop the class labels from the original data
        train_labels = train_data.pop("class").astype("str").map(self._attack_map())
        test_labels = test_data.pop("class").astype("str").map(self._attack_map())

        # Finally, 1-Hot encode the categorical data
        train_data = pd.get_dummies(train_data)
        test_data = pd.get_dummies(test_data)
        assert (train_data.columns == test_data.columns).all()

        # We'll use ndarrays from now on
        super(NSL_KDD, self).__init__(
            x_train=train_data.to_numpy(), y_train=train_labels.to_numpy(),
            x_test=test_data.to_numpy(), y_test=test_labels.to_numpy(), *args, **kwargs
        )

    def _attack_map(self) -> dict:
        """
        Map grouping the single attack classes
        :return: mapping dictionary
        """

        attack_dict = {
            'normal': 'normal',

            'back': 'DoS',
            'land': 'DoS',
            'neptune': 'DoS',
            'pod': 'DoS',
            'smurf': 'DoS',
            'teardrop': 'DoS',
            'mailbomb': 'DoS',
            'apache2': 'DoS',
            'processtable': 'DoS',
            'udpstorm': 'DoS',

            'ipsweep': 'Probe',
            'nmap': 'Probe',
            'portsweep': 'Probe',
            'satan': 'Probe',
            'mscan': 'Probe',
            'saint': 'Probe',

            'ftp_write': 'R2L',
            'guess_passwd': 'R2L',
            'imap': 'R2L',
            'multihop': 'R2L',
            'phf': 'R2L',
            'spy': 'R2L',
            'warezclient': 'R2L',
            'warezmaster': 'R2L',
            'sendmail': 'R2L',
            'named': 'R2L',
            'snmpgetattack': 'R2L',
            'snmpguess': 'R2L',
            'xlock': 'R2L',
            'xsnoop': 'R2L',
            'worm': 'R2L',

            'buffer_overflow': 'U2R',
            'loadmodule': 'U2R',
            'perl': 'U2R',
            'rootkit': 'U2R',
            'httptunnel': 'U2R',
            'ps': 'U2R',
            'sqlattack': 'U2R',
            'xterm': 'U2R'
        }

        return attack_dict

    def _preprocess(self):
        """
        Minmaxscale the data
        :return:
        """
        self.scikit_scale(scikit_scaler=MinMaxScaler)


class IDS(DataLabels):
    def __init__(
            self, start: int = None, stop: int = None,
            data_path: Path=(BASE_PATH / "data" / "IDS" / "ids").with_suffix(".h5"), *args, **kwargs
    ):
        """
        IDS data set: https://www.unb.ca/cic/datasets/ids-2018.html
        Download the data using awscli:
        aws s3 sync --no-sign-request --region eu-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" <dst-folder>
        :param start: first row number to read (for debugging)
        :param stop: last row number to read (for debugging)
        :param data_path: file with preprocessed data
        """

        # Open data
        if data_path.exists():
            df = pd.read_hdf(data_path, start=start, stop=stop)
        else:
            df = self.csv_to_h5(data_path=data_path)

        # Map the labels accordingly
        df["Label"] = df["Label"].map(self._attack_map())
        labels = df.pop("Label")

        # drop some object features if exist
        df = df.drop(columns=["Src IP", "Flow ID", "Dst IP"], errors="ignore")

        super(IDS, self).__init__(x_train=df.to_numpy(), y_train=labels.to_numpy(), *args, **kwargs)

    def _attack_map(self) -> dict:
        """
        Map grouping the single attack classes
        :return: mapping dictionary
        """

        attack_dict = {
            'Benign': 'Benign',

            'Bot': 'Bot',

            'FTP-BruteForce': 'BruteForce',
            'SSH-Bruteforce': 'BruteForce',

            'DDoS attacks-LOIC-HTTP': 'DDoS',
            'DDOS attack-LOIC-UDP': 'DDoS',
            'DDOS attack-HOIC': 'DDoS',

            'DoS attacks-GoldenEye': 'DoS',
            'DoS attacks-Slowloris': 'DoS',
            'DoS attacks-SlowHTTPTest': 'DoS',
            'DoS attacks-Hulk': 'DoS',

            'Infilteration': 'Infiltration',

            'Brute Force -Web': 'WebAttacks',
            'Brute Force -XSS': 'WebAttacks',
            'SQL Injection': 'WebAttacks',
        }

        return attack_dict

    def _type_map(self):

        return {
            "Dst Port": "integer",
            "Protocol": "integer",
            "Flow Duration": "integer",
            "Tot Fwd Pkts": "integer",
            "Tot Bwd Pkts": "integer",
            "TotLen Fwd Pkts": "integer",
            "TotLen Bwd Pkts": "integer",
            "Fwd Pkt Len Max": "integer",
            "Fwd Pkt Len Min": "integer",
            "Fwd Pkt Len Mean": "float",
            "Fwd Pkt Len Std": "float",
            "Bwd Pkt Len Max": "integer",
            "Bwd Pkt Len Min": "integer",
            "Bwd Pkt Len Mean": "float",
            "Bwd Pkt Len Std": "float",
            "Flow Byts/s": "float",
            "Flow Pkts/s": "float",
            "Flow IAT Mean": "float",
            "Flow IAT Std": "float",
            "Flow IAT Max": "integer",
            "Flow IAT Min": "integer",
            "Fwd IAT Tot": "integer",
            "Fwd IAT Mean": "float",
            "Fwd IAT Std": "float",
            "Fwd IAT Max": "integer",
            "Fwd IAT Min": "integer",
            "Bwd IAT Tot": "integer",
            "Bwd IAT Mean": "float",
            "Bwd IAT Std": "float",
            "Bwd IAT Max": "integer",
            "Bwd IAT Min": "integer",
            "Fwd PSH Flags": "integer",
            "Bwd PSH Flags": "integer",
            "Fwd URG Flags": "integer",
            "Bwd URG Flags": "integer",
            "Fwd Header Len": "integer",
            "Bwd Header Len": "integer",
            "Fwd Pkts/s": "float",
            "Bwd Pkts/s": "float",
            "Pkt Len Min": "integer",
            "Pkt Len Max": "integer",
            "Pkt Len Mean": "float",
            "Pkt Len Std": "float",
            "Pkt Len Var": "float",
            "FIN Flag Cnt": "integer",
            "SYN Flag Cnt": "integer",
            "RST Flag Cnt": "integer",
            "PSH Flag Cnt": "integer",
            "ACK Flag Cnt": "integer",
            "URG Flag Cnt": "integer",
            "CWE Flag Count": "integer",
            "ECE Flag Cnt": "integer",
            "Down/Up Ratio": "integer",
            "Pkt Size Avg": "float",
            "Fwd Seg Size Avg": "float",
            "Bwd Seg Size Avg": "float",
            "Fwd Byts/b Avg": "integer",
            "Fwd Pkts/b Avg": "integer",
            "Fwd Blk Rate Avg": "integer",
            "Bwd Byts/b Avg": "integer",
            "Bwd Pkts/b Avg": "integer",
            "Bwd Blk Rate Avg": "integer",
            "Subflow Fwd Pkts": "integer",
            "Subflow Fwd Byts": "integer",
            "Subflow Bwd Pkts": "integer",
            "Subflow Bwd Byts": "integer",
            "Init Fwd Win Byts": "integer",
            "Init Bwd Win Byts": "integer",
            "Fwd Act Data Pkts": "integer",
            "Fwd Seg Size Min": "integer",
            "Active Mean": "float",
            "Active Std": "float",
            "Active Max": "integer",
            "Active Min": "integer",
            "Idle Mean": "float",
            "Idle Std": "float",
            "Idle Max": "integer",
            "Idle Min": "integer"
        }

    def csv_to_h5(self, data_path: Path):
        """
        Open raw data, preprocess and save in single file.
        Note: all NaN & infinity rows are dropped.
        :param data_path: path to the data
        """
        print("We need to convert the raw data first. This might take some time.")

        # Look for all suitable raw files
        all_files = [cur_file for cur_file in data_path.parent.iterdir() if cur_file.suffix == ".csv"]

        # Combine to overall data
        all_df = pd.DataFrame()
        for i_file, cur_file in enumerate(all_files):
            # Open the respective file
            cur_df = pd.read_csv(
                cur_file, header=0, parse_dates=["Timestamp"], index_col=["Timestamp"], low_memory=False, na_values="Infinity"
            )

            # For whatever reason, they repeat the header row within one csv file. Drop these.
            try:
                cur_df = cur_df.drop(index="Timestamp", errors="ignore")
            except TypeError:
                pass

            # Drop rows with NaN, infinity
            cur_df = cur_df.dropna()

            # Convert remaining types automatically; infer_object() only returns objects
            type_map = self._type_map()
            for cur_col in cur_df.columns:
                if cur_col in type_map:
                    cur_df[cur_col] = pd.to_numeric(cur_df[cur_col], downcast=type_map[cur_col])

            all_df = pd.concat([all_df, cur_df], sort=False)

        # For whatever reason, there is not always a source port
        try:
            all_df["Src Port"] = pd.to_numeric(all_df["Src Port"], downcast="unsigned")
        except KeyError:
            pass
        # Category type also saves some space
        all_df["Protocol"] = all_df["Protocol"].astype("category")
        # One-hot encoding
        all_df = pd.get_dummies(all_df, columns=['Protocol'])

        # Save and return
        all_df.reset_index(drop=True).to_hdf(data_path, key="ids", format="table")
        return all_df

    def _preprocess(self):
        """
        Minmaxscale the data
        :return:
        """

        self.scikit_scale(scikit_scaler=MinMaxScaler)

