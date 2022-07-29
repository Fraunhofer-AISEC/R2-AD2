This project is not maintained.
It has been published as part of the following conference paper at ECML-PKDD 2022:
# R2-AD2: Detecting Anomalies by Analysing the Raw Gradient 
## by Jan-Philipp Schulze, Philip Sperl, Ana Răduțoiu, Carla Sagebiel and Konstantin Böttinger

Neural networks follow a gradient-based learning scheme, adapting their mapping parameters by back-propagating the output loss.
Samples unlike the ones seen during training cause a different gradient distribution.
Based on this intuition, we design a novel semi-supervised anomaly detection method called R2-AD2.
By analysing the temporal distribution of the gradient over multiple training steps, we reliably detect point anomalies in strict semi-supervised settings.
Instead of domain dependent features, we input the raw gradient caused by the sample under test to an end-to-end recurrent neural network architecture.
R2-AD2 works in a purely data-driven way, thus is readily applicable in a variety of important use cases of anomaly detection.

### Dependencies
We used ``docker`` during the development.
You can recreate our environment by:  
``docker build -t r2ad2 ./docker/``.

Afterwards, start an interactive session while mapping the source folder in the container:  
``docker run --gpus 1 -it --rm -v ~/path/to/r2ad2/:/app/ -v ~/.keras:/root/.keras r2ad2``

#### Data sets
The raw data sets are stored in ``./data/``.
You need to add Darknet [1], CovType [2], CreditCard[3], DoH [4], IDS [5], URL [6], Mammography [7], NSL-KDD [8] and URL [9] from the respective website.

For example, the URL's archive contains the file ``All.csv``.
Move it to ``./data/url/All.csv``.
The rest is automatically handled in ``./libs/DataHandler.py``, where you find more information which file is loaded.

#### Baseline methods
The baseline methods are stored in ``./baselines/``.
Whereas we implemented A3, GradCon and Deep-SAD, you need to add DevNet [10] manually from their website.

### Instructions

#### Train models
For each data set, all applicable experiments are bundled in the respective ``do_*.py``.
You need to provide a random seed and whether the results should be evaluated on the "val" or "test" data, e.g. ``python3 ./do_mnist.py 123 val``.
Optional arguments are e.g. the training data pollution ``--p_contamination`` and the number of known anomalies ``--n_train_anomalies``.
Please note that we trained the models on a GPU, i.e. there will still be randomness while training the models.
Your models are stored in ``./models/`` if not specified otherwise using ``--model_path``.

#### Evaluate models
After training, the respective models are automatically evaluated on the given data split.
As output, a ``.metric.csv``, ``.roc.csv`` and ``.roc.png`` are given.
By default, these files are stored in ``./models/{random_seed}/``.
The test results are the merged results of 5 runs (using the random seeds 110, 210, 310, 410 and 510), e.g.: 

``python3 evaluate_results.py 110 210 310 410 510 --p_pollution 0.0 --n_anomalies 100 --p_name wilcoxon --show_stddev 1``

### Known Limitations
For the CNN-based targets (MNIST & FMNIST), TensorFlow throws an error (``Conv2DBackpropFilter uses a while_loop. Fix that!``).
Training works fine, but is not parallelised thus runs very slowly.
There are already some bug reports on TensorFlow's Github, so we hope the error is fixed in future.

We sometimes had problems loading the trained models in TensorFlow's eager mode.
Please use graph mode instead.

### File Structure
```
R2-AD2
│   do_*.py                     (start experiment on the respective data set)
│   evaluate_results.py         (calculate the mean over the test results)
│   README.md                   (file you're looking at)
│
└─── data                       (raw data)
│
└─── docker                     (folder for the Dockerfile)
│   │   requirements.txt        (dependencies in case you don't want to use docker)
│
└─── libs
│   └───architecture            (network architecture of the alarm and target networks)
│   └───network                 (helper functions for the NNs)
│   │   constants.py            (default arguments while training)
│   │   GAA.py                  (main library for our anomaly detection method - we came up with the extra punny acronym "R2-AD2" later in development)
│   │   DataHandler.py          (reads, splits, and manages the respective data set)
│   │   ExperimentWrapper.py    (wrapper class to generate comparable experiments)
│   │   Metrics.py              (methods to evaluate the data)
│
└─── models                     (output folder for the trained neural networks)
│
└─── baselines                  (baseline methods)
│
```

### Links
* [1] https://www.unb.ca/cic/datasets/darknet2020.html
* [2] https://archive.ics.uci.edu/ml/datasets/Covertype
* [3] https://www.kaggle.com/mlg-ulb/creditcardfraud
* [4] https://www.unb.ca/cic/datasets/dohbrw-2020.html
* [5] https://www.unb.ca/cic/datasets/ids-2018.html
* [6] https://www.unb.ca/cic/datasets/url-2016.html
* [7] http://odds.cs.stonybrook.edu/mammography-dataset/
* [8] https://www.unb.ca/cic/datasets/nsl.html
* [9] https://www.unb.ca/cic/datasets/url-2016.html
* [10] https://github.com/GuansongPang/deviation-network

