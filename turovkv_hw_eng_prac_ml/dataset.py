import os
import random
import sys

import numpy as np
import pandas
from sklearn.model_selection import train_test_split


def read_dataset(path):
    dataframe = pandas.read_csv(path, header=1)
    dataset = dataframe.values.tolist()
    random.shuffle(dataset)
    y = [row[0] for row in dataset]
    X = [row[1:] for row in dataset]
    return np.array(X), np.array(y)


def main(input, output):
    X, y = read_dataset(input)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    os.makedirs(output, exist_ok=True)
    np.save(f"{output}/X_train", X_train)
    np.save(f"{output}/X_test", X_test)
    np.save(f"{output}/y_train", y_train)
    np.save(f"{output}/y_test", y_test)


if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    main(input, output)
