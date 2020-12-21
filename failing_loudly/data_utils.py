# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np

# -------------------------------------------------
# DATA UTILS
# -------------------------------------------------


def __unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def normalize_datapoints(x, factor):
    x = x.astype('float32') / factor
    return x


def random_shuffle(x, y):
    x, y = __unison_shuffled_copies(x, y)
    return x, y


def random_shuffle_and_split(x_train, y_train, x_test, y_test, split_index):
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    x, y = __unison_shuffled_copies(x, y)

    x_train = x[:split_index, :]
    x_test = x[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return (x_train, y_train), (x_test, y_test)


def random_set_features_to_uniform(x, corruption_probability=0.2):
    p = [corruption_probability, 1 - corruption_probability]
    (n_samples, n_features) = x.shape
    corrupt_decisions = np.random.choice(['to_corrupt', 'ok'], size=n_samples, p=p)
    corrupt_indices = np.where(corrupt_decisions == 'to_corrupt')[0]
    n_samples_corrupt = corrupt_indices.shape[0]
    p_uniform = 1. / n_features * np.ones((n_features,))
    feature_decisions = np.random.choice(list(range(n_features)), size=n_samples_corrupt, p=p_uniform)
    x[corrupt_indices, :] = np.zeros((n_samples_corrupt, n_features))
    x[tuple([corrupt_indices, feature_decisions])] = 1.
    return x

