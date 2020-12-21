from sklearn import preprocessing
import numpy as np
from .preprocessing import Preprocessor
import os
import pickle
import pandas as pd


def get_X_y(df, target):
    X = np.array(df.loc[:, df.columns != target])
    y = np.array(df.loc[:, df.columns == target])
    return X, y


def sample_df_dataset(df_train_name, df_valid_name, df_test_name, run_seed=1234, max_num_row=10000):
    # always same samples here
    np.random.seed(1234)
    df_train = pd.read_csv(df_train_name)
    df_train = df_train.sample(n=min(max_num_row, df_train.shape[0]), random_state=1234)

    np.random.seed(run_seed)
    df_valid = pd.read_csv(df_valid_name)
    df_valid = df_valid.sample(n=min(max_num_row, df_valid.shape[0]), random_state=run_seed)
    df_test = pd.read_csv(df_test_name)
    df_test = df_test.sample(n=min(max_num_row, df_test.shape[0]), random_state=run_seed)

    return df_train, df_valid, df_test


def get_prepared_dataset(df_train, df_valid, df_test, target):
    X_tr_orig, y_tr_orig = get_X_y(df_train, target)
    X_val_orig, y_val_orig = get_X_y(df_valid, target)
    X_te_orig, y_te_orig = get_X_y(df_test, target)

    orig_dims = X_tr_orig.shape[1]

    le = preprocessing.LabelEncoder()
    y_tr_orig = le.fit_transform(y_tr_orig)

    y_val_orig = le.transform(y_val_orig)
    y_te_orig = le.transform(y_te_orig)

    nb_classes = le.classes_.shape[0]

    return (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes, le


def get_name(proc_fname):
    els = proc_fname.split(':')
    if len(els) > 1:
        return els[1]
    else:
        return proc_fname


def is_in_group(fname, categorical):
    return np.any([f == get_name(fname) for f in categorical])


def get_group(fname, categorical):
    return [f for f in categorical if f == get_name(fname)]


def get_rand_dataset_split(df_train_name, df_valid_name, df_test_name, target, max_num_row, rand_run,
                           out_path, save_dataset=True):
    print("Loading data...")
    # Load data.

    train_filename = f'{out_path}/train_dataframe.pkl'
    valid_test_filename = f'{out_path}/valid_test_dataframes_{rand_run}.pkl'
    processed_data_filename = f'{out_path}/processed_data_{rand_run}.pkl'

    if os.path.exists(train_filename):
        print("Loading from file: ", train_filename)
        with open(train_filename, 'rb') as f:
            [df_train, target] = pickle.load(f)
    else:
        df_train, df_valid, df_test = sample_df_dataset(
            df_train_name, df_valid_name, df_test_name, run_seed=rand_run, max_num_row=max_num_row)

        if save_dataset:
            with open(train_filename, 'wb') as f:
                pickle.dump([df_train, target], f)

    if os.path.exists(valid_test_filename):
        print("Loading from file: ", valid_test_filename)
        with open(valid_test_filename, 'rb') as f:
            [df_valid, df_test] = pickle.load(f)
    else:
        _, df_valid, df_test = sample_df_dataset(
            df_train_name, df_valid_name, df_test_name, run_seed=rand_run, max_num_row=max_num_row)

        if save_dataset:
            with open(valid_test_filename, 'wb') as f:
                pickle.dump([df_valid, df_test], f)

    dataframes = [df_train, df_valid, df_test]

    # process data

    prep = Preprocessor(df_train, target)
    df_train_proc = prep.get_processed_df(df_test=None)
    df_valid_proc = prep.get_processed_df(df_test=df_valid)
    df_test_proc = prep.get_processed_df(df_test=df_test)

    dataframes.extend([df_test_proc.columns, df_test_proc.dtypes])

    feature_names = list(df_train_proc.columns)
    feature_names.remove(target)

    if os.path.exists(processed_data_filename):
        print("Loading from file: ", processed_data_filename)
        with open(processed_data_filename, 'rb') as f:
            processed_data = pickle.load(f)
            try:
                [(X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes, le,
                 feature_names] = processed_data
            except ValueError as e:
                print(e)
                print('Do not load feature names')
                print(feature_names)
                [(X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes,
                 le] = processed_data
    else:
        (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes, le = \
            get_prepared_dataset(df_train_proc, df_valid_proc, df_test_proc, target)

        processed_data = [(X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims,
                          nb_classes, le, feature_names]
        if save_dataset:
            with open(processed_data_filename, 'wb') as f:
                pickle.dump(processed_data, f)

    print("%%%%%%%%%%%%%%%%% Summary")
    print(X_tr_orig.shape)
    print(np.unique(y_tr_orig, return_counts=True))
    print(X_val_orig.shape)
    print(np.unique(y_val_orig, return_counts=True))
    print(X_te_orig.shape)
    print(np.unique(y_te_orig, return_counts=True))

    categorical = prep._get_categorical_features()
    numerical = prep._get_numerical_features()

    features = list(df_valid_proc.columns)
    features.remove(target)
    numerical_features = [f for f in range(X_tr_orig.shape[1]) if is_in_group(features[f], numerical)]
    categorical_features = [f for f in range(X_tr_orig.shape[1]) if is_in_group(features[f], categorical)]
    categorical_groups = dict()
    for f in categorical_features:
        group = get_group(features[f], categorical)[0]
        if group in categorical_groups:
            categorical_groups[group].append(f)
        else:
            categorical_groups[group] = [f]
    categorical_groups = list(categorical_groups.values())

    return dataframes, processed_data, numerical_features, categorical_groups
