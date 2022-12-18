#!/usr/bin/env python
# coding: utf-8
import sys

import numpy as np
from tree_classifier import DecisionTreeClassifier

# train.py
from dvclive import Live


def main(input):
    dtc = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)

    X_train = np.load(f"{input}/X_train.npy")
    X_test = np.load(f"{input}/X_test.npy")
    y_train = np.load(f"{input}/y_train.npy")
    y_test = np.load(f"{input}/y_test.npy")

    dtc.fit(X_train, y_train)
    y_p = dtc.predict(X_test)
    correct = np.nonzero(y_p == y_test)[0].shape[0]
    all = y_test.shape[0]

    with Live() as live:
        live.log_param("max_depth", 6)
        live.log_param("min_samples_leaf", 5)
        live.log_metric("accuracy", correct / all)
    print(f"accuracy = {correct / all}")


if __name__ == "__main__":
    input = sys.argv[1]
    main(input)
