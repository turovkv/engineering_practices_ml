#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from tree_classifier.dataset import read_dataset
from tree_classifier.tree_classifier import DecisionTreeClassifier
from tree_classifier.visualisation import draw_tree, plot_2d, plot_roc_curve

# -------------------------------------------------------------------------
# ---- test data ---------


def main(path="train.csv"):
    noise = 0.35
    X, y = make_moons(1500, noise=noise)
    X_test, y_test = make_moons(200, noise=noise)
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
    tree.fit(X, y)
    plot_2d(tree, X, y)
    plot_roc_curve(y_test, tree.predict_proba(X_test))
    draw_tree(tree)

    X, y = make_blobs(1500, 2, centers=[[0, 0], [-2.5, 0], [3, 2], [1.5, -2.0]])
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
    tree.fit(X, y)
    plot_2d(tree, X, y)
    draw_tree(tree)

    # -------------------------------------------------------------------------
    # ---- real data ---------

    X, y = read_dataset(path)
    dtc = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    dtc.fit(X_train, y_train)
    y_p = dtc.predict(X_test)
    correct = np.nonzero(y_p == y_test)[0].shape[0]
    all = y_test.shape[0]
    print(f"accuracy = {correct / all}")
