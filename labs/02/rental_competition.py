#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        data, target = train.data, train.target

        categorical = list(range(0, 8))
        numeric = list(range(8,12))

        def make_model(degree, alpha, interact):
            """
            Creates a regression pipeline. Preprocesses the data and adds L2 regularization.
            :param degree: polynomial degree for numerical features
            :param alpha: hyperparameter for L2 regularization
            :param interact: add interaction terms (including categorical features)
            :return: pipeline
            """

            num = sklearn.pipeline.Pipeline([
                ("scale1", sklearn.preprocessing.StandardScaler()),
                ("poly", sklearn.preprocessing.PolynomialFeatures(degree, include_bias=False)),
                ("scale2", sklearn.preprocessing.StandardScaler())
            ])

            columns = sklearn.compose.ColumnTransformer([
                ("cat", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical),
                ("num", num, numeric)
            ])

            steps = [("columns", columns)]

            if interact:
                steps += [("interact", sklearn.preprocessing.PolynomialFeatures(2, include_bias=False, interaction_only=True)),
                          ("scale3", sklearn.preprocessing.StandardScaler())]

            steps.append(("ridge",  sklearn.linear_model.Ridge(alpha=alpha)))

            return sklearn.pipeline.Pipeline(steps)

        kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=args.seed)

        def cv_rmse(degree, alpha, interact):
            """
            average RMSE for given hyperparameters
            """
            scores = sklearn.model_selection.cross_val_score(
                make_model(degree, alpha, interact), data, target, cv=kfold, scoring="neg_root_mean_squared_error"
            )
            return -scores.mean()

        best_rmse, best_params = np.inf, None

        for interact in [False, True]:
            for degree in range(1, 4):
                for alpha in np.logspace(-3, 4, 8):
                    rmse = cv_rmse(degree, alpha, interact)
                    if rmse < best_rmse:
                        best_rmse, best_params = rmse, (degree, alpha, interact)

        degree, best_alpha, interact = best_params

        for alpha in np.logspace(np.log10(best_alpha) - 0.5, np.log10(best_alpha) + 0.5, 9):
            rmse = cv_rmse(degree, alpha, interact)
            if rmse < best_rmse:
                best_rmse, best_params = rmse, (degree, alpha, interact)

        model = make_model(*best_params)
        model.fit(data, target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = np.maximum(model.predict(test.data), 0)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
