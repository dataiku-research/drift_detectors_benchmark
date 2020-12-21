# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from math import ceil

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

from sklearn.ensemble import RandomForestClassifier

from art.attacks import BoundaryAttack, ZooAttack, HopSkipJump
from art.classifiers import SklearnClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# CONCEPT SHIFT

def any_other_label(label_to_change, labels):
    if len(labels) == 1:
        #print('Warning: no other possible value to select. Returning the one single allowed value.')
        return label_to_change
    allowed_indices = np.where(labels != label_to_change)[0]
    new_label = labels[np.random.choice(allowed_indices)]
    return new_label


# Random change class of a subset of data.
def random_class_subset_shift(x, y, perc_max_changes=0.3):
    labels, counts = np.unique(y, return_counts=True)

    subset_indices = np.random.choice(x.shape[0], ceil(x.shape[0] * perc_max_changes), replace=False)

    n_max_changes = int(np.floor(perc_max_changes * y.shape[0]))
    subset_indices = subset_indices[:n_max_changes]

    rand_labeler = lambda y: any_other_label(y, labels)
    vec_rand_labeler = np.vectorize(rand_labeler)
    y[subset_indices] = vec_rand_labeler(y[subset_indices])

    return x, y, subset_indices


# PRIOR SHIFT


# Resample instances of all classes by given priors.
def rebalance_shift(x, y, priors):
    labels, counts = np.unique(y, return_counts=True)
    n_labels = len(labels)
    n_priors = len(priors)
    assert (n_labels == n_priors)
    assert (np.sum(priors) == 1.)
    n_to_sample = y.shape[0]
    for label_idx, prior in enumerate(priors):
        if prior > 0:
            n_samples_label = counts[label_idx]
            max_n_to_sample = np.round(n_samples_label / prior)
            if n_to_sample > max_n_to_sample:
                n_to_sample = max_n_to_sample

    resampled_counts = [int(np.round(prior * n_to_sample)) for prior in priors]
    resampled_indices = []
    for cl, res_count in zip(labels, resampled_counts):
        if res_count:
            cl_indices = np.where(y == cl)[0]
            cl_res_indices = np.random.choice(cl_indices, res_count, replace=False)
            resampled_indices.extend(cl_res_indices)

    x = x[resampled_indices, :]
    y = y[resampled_indices]
    return x, y


# Remove instances of a single class.
def knockout_shift(x, y, cl, delta):
    del_indices = np.where(y == cl)[0]
    until_index = ceil(delta * len(del_indices))
    if until_index % 2 != 0:
        until_index = until_index + 1
    del_indices = del_indices[:until_index]
    x = np.delete(x, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    return x, y


# Remove all classes except for one via multiple knock-out.
def only_one_shift(x, y, keep_cl):
    labels = np.unique(y)
    for cl in labels:
        if cl != keep_cl:
            x, y = knockout_shift(x, y, cl, 1.0)

    return x, y


# COVARIATE SHIFT

# Keeps an observation with a probability which decreases as points are further away from the samples mean
# gamma is the fraction of samples close to the mean we want to keep
def subsample_joint_shift(x, y, gamma=0.8, shift_features=None):
    if shift_features is None:
        shift_features = list(range(x.shape[1]))

    x_mean = np.mean(x[:, shift_features], axis=0)
    distance = np.sqrt(np.sum((x[:, shift_features] - x_mean) ** 2, axis=1))
    gamma_quantile = np.quantile(distance, gamma)
    ln_prob_keep_far = np.log(0.5)  # sample with probability 50% samples with distance after gamma quantile
    probabilities = np.exp(ln_prob_keep_far / gamma_quantile * distance)
    keep_decisions = np.array([np.random.choice(['keep', 'remove'], size=1, p=[p, 1 - p])[0] for p in probabilities])
    keep_indices = np.where(keep_decisions == 'keep')[0]

    x = x[keep_indices, :]
    y = y[keep_indices]
    return x, y, shift_features


def is_integer_feature(x, f):
    values = np.unique(x[:, f])
    if values.dtype.char in np.typecodes['AllInteger']:
        return True
    else:
        return np.all([v.is_integer() for v in values])


def is_categorical_feature(x, f):
    values = np.unique(x[:, f])
    n_values = len(values)
    is_categorical = is_integer_feature(x, f) & (n_values < 1000)
    return is_categorical


def get_feature_split_value(x, f):
    #values = np.unique(x[:, f])
    values = x[:, f]
    f_split = np.median(values)
    return f_split


# Subsample feature f with probability p when f<=f_split and 1-p when f>f_split.
def subsample_one_feature_shift(x, y, f, f_split, p=0.5, one_side=True, min_size=5000):
    smaller_than_split_indices = np.where(x[:, f] <= f_split)[0]
    n_smaller = len(smaller_than_split_indices)
    larger_than_split_indices = np.where(x[:, f] > f_split)[0]
    n_larger = len(larger_than_split_indices)

    keep_smaller_decisions = np.random.choice(['keep', 'remove'], size=n_smaller, p=[p, 1 - p])
    keep_smaller_indices = smaller_than_split_indices[np.where(keep_smaller_decisions == 'keep')[0]]

    if one_side:
        keep_larger_indices = larger_than_split_indices
    else:
        keep_larger_decisions = np.random.choice(['keep', 'remove'], size=n_larger, p=[1 - p, p])
        keep_larger_indices = larger_than_split_indices[np.where(keep_larger_decisions == 'keep')[0]]

    keep_indices = np.hstack([keep_smaller_indices, keep_larger_indices])
    if len(keep_indices) < min_size:
        to_add = min_size - len(keep_indices)
        remove_smaller_indices = smaller_than_split_indices[np.where(keep_smaller_decisions == 'remove')[0]]
        keep_indices = np.hstack([keep_smaller_indices, remove_smaller_indices[:to_add], keep_larger_indices])

    x = x[keep_indices, :]
    y = y[keep_indices]
    return x, y


# Subsample all features with probability p when f<=f_split and 1-p when f>f_split
def subsample_feature_shift(x, y, feat_delta=0.5, p=0.5, numerical_features=None, one_side=True, min_size=5000):

    if numerical_features is not None:
        n_numerical = len(numerical_features)
        n_feat_subsample = max(1, ceil(n_numerical * feat_delta))
        feat_indices = np.random.choice(n_numerical, n_feat_subsample, replace=False)
        feat_indices = np.array(numerical_features)[feat_indices]
    else:
        n_feat_subsample = min(1, ceil(x.shape[1] * feat_delta))
        feat_indices = np.random.choice(x.shape[1], n_feat_subsample, replace=False)

    for f in feat_indices:
        f_split = get_feature_split_value(x, f)
        x, y = subsample_one_feature_shift(x, y, f, f_split, p=p, one_side=one_side, min_size=min_size)
    return x, y, feat_indices


def split_categorical_feature(x, group_features):
    unique_rows = np.unique(x[:, np.array(group_features)], axis=0)
    n_half = ceil(0.5 * unique_rows.shape[0])
    choice = np.random.choice(range(unique_rows.shape[0]), size=(n_half,), replace=False)
    half_1 = np.zeros(unique_rows.shape[0], dtype=bool)
    half_1[choice] = True
    half_2 = ~half_1
    return unique_rows[half_1], unique_rows[half_2]


# Subsample categorical feature with probability p when f is in a random selection of categories,
# 1-p when f is in the remaining categories.
def subsample_categorical_feature_shift(x, y, group_features, p=0.5, return_groups=False, one_side=True, min_size=5000):
    group_1, group_2 = split_categorical_feature(x, group_features)
    smaller_than_split_indices = np.where(
        np.apply_along_axis(lambda val: (val == group_1).all(axis=1).any(), 1, x[:, group_features]))[0]
    n_smaller = len(smaller_than_split_indices)
    larger_than_split_indices = np.where(
        np.apply_along_axis(lambda val: (val == group_2).all(axis=1).any(), 1, x[:, group_features]))[0]
    n_larger = len(larger_than_split_indices)

    keep_smaller_decisions = np.random.choice(['keep', 'remove'], size=n_smaller, p=[p, 1 - p])
    keep_smaller_indices = smaller_than_split_indices[np.where(keep_smaller_decisions == 'keep')[0]]

    if one_side:
        keep_larger_indices = larger_than_split_indices
    else:
        keep_larger_decisions = np.random.choice(['keep', 'remove'], size=n_larger, p=[1 - p, p])
        keep_larger_indices = larger_than_split_indices[np.where(keep_larger_decisions == 'keep')[0]]

    keep_indices = np.hstack([keep_smaller_indices, keep_larger_indices])
    if len(keep_indices) < min_size:
        to_add = min_size - len(keep_indices)
        remove_smaller_indices = smaller_than_split_indices[np.where(keep_smaller_decisions == 'remove')[0]]
        keep_indices = np.hstack([keep_smaller_indices, remove_smaller_indices[:to_add], keep_larger_indices])

    x = x[keep_indices, :]
    y = y[keep_indices]

    if not return_groups:
        return x, y
    else:
        return x, y, group_1, group_2


def subsample_all_categorical_feature_shift(x, y, categorical_groups, p=0.5, min_size=5000):
    for group_features in categorical_groups:
        x, y = subsample_categorical_feature_shift(x, y, group_features, p=p, min_size=min_size)
    return x, y


# Subsample all features with probability p when f<=f_split and 1-p when f>f_split
def subsample_all_feature_shift(x, y, p=0.5, numerical_features=None, categorical_groups=None, one_side=True, min_size=5000):
    x, y, _ = subsample_feature_shift(x, y, feat_delta=1.0, p=p, numerical_features=numerical_features,
                                      one_side=one_side, min_size=min_size)
    if categorical_groups:
        x, y = subsample_all_categorical_feature_shift(x, y, categorical_groups, p=p, min_size=min_size)

    return x, y


# Gaussian Noise applied on delta portion of samples and feat_delta portion of features
def gaussian_noise_shift(x, y, noise_amt=10., delta=1.0, feat_delta=1.0,
                         numerical_features=None, clip=True, ceil_int=True):
    x, indices, feat_indices = gaussian_noise_subset(
        x, noise_amt, normalization=1.0, delta=delta, feat_delta=feat_delta,
        numerical_features=numerical_features, clip=clip, ceil_int=ceil_int)
    return x, y, indices, feat_indices


def gaussian_noise(x, noise_amt, normalization=1.0, clip=True):
    noise = np.random.normal(0, noise_amt / normalization, (x.shape[0], x.shape[1]))
    if clip:
        x_mins = np.min(x, axis=0)
        x_maxs = np.max(x, axis=0)
        x_clipped = np.clip(x + noise, x_mins, x_maxs)
        return x_clipped
    else:
        return x + noise


def gaussian_noise_subset(x, noise_amt, normalization=1.0, delta=1.0, feat_delta=1.0,
                          numerical_features=None, clip=True, ceil_int=True):
    # precompute for clip and ceil
    int_features = [f for f in range(x.shape[1]) if is_integer_feature(x, f)]
    x_mins = np.min(x, axis=0)
    x_maxs = np.max(x, axis=0)

    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
    indices = np.transpose(indices[np.newaxis])

    if numerical_features is not None:
        n_numerical = len(numerical_features)
        feat_indices = np.random.choice(n_numerical, ceil(n_numerical * feat_delta), replace=False)
        feat_indices = np.array(numerical_features)[feat_indices]
    else:
        feat_indices = np.random.choice(x.shape[1], ceil(x.shape[1] * feat_delta), replace=False)
    feat_indices = feat_indices[np.newaxis]

    x_mod = x[indices, feat_indices]
    noise = np.random.normal(0, noise_amt / normalization, (x_mod.shape[0], x_mod.shape[1]))
    x_mod = x_mod + noise

    if bool(int_features) & ceil_int:
        int_features = list(set(int_features) & set(feat_indices.flatten()))
        int_features_mapped = [list(feat_indices.flatten()).index(f_idx) for f_idx in int_features]
        x_mod[:, int_features_mapped] = np.ceil(x_mod[:, int_features_mapped])

    if clip:
        x_mod = np.clip(x_mod, x_mins[np.squeeze(feat_indices)], x_maxs[np.squeeze(feat_indices)])

    x[indices, feat_indices] = x_mod
    return x, indices, feat_indices.flatten()


# Gaussian Noise applied on delta portion of samples and feat_delta portion of features
def constant_value_shift(x, y, delta=1.0, feat_delta=1.0, numerical_features=None):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
    indices = np.transpose(indices[np.newaxis])

    if numerical_features is not None:
        n_numerical = len(numerical_features)
        feat_indices = np.random.choice(n_numerical, ceil(n_numerical * feat_delta), replace=False)
        feat_indices = np.array(numerical_features)[feat_indices]
    else:
        feat_indices = np.random.choice(x.shape[1], ceil(x.shape[1] * feat_delta), replace=False)
    feat_indices = feat_indices[np.newaxis]

    x_mod = x[indices, feat_indices]
    med = np.median(x[:, feat_indices], axis=0)
    constant_val = np.repeat(med, x_mod.shape[0], axis=0)

    x[indices, feat_indices] = constant_val

    return x, y, indices.flatten(), feat_indices.flatten()


# Set a fraction of samples and features to median constant values (numerical) or random constant category (categorical)
def constant_all_feature_shift(x, y, delta=1.0, feat_delta=1.0, numerical_features=None, categorical_groups=None):
    x, y, indices, features = constant_value_shift(x, y, delta=delta, feat_delta=feat_delta,
                                                   numerical_features=numerical_features)
    if categorical_groups:
        x[indices, :], y[indices], _, cat_features = constant_categorical_shift(x[indices, :], y[indices],
                                                                                   delta=1.0, feat_delta=feat_delta,
                                                                                   categorical_groups=categorical_groups)
        features = np.concatenate((features, cat_features))

    return x, y, indices, features


def any_other_array(array_to_change, arrays):
    if len(arrays) == 1:
        # print('Warning: no other possible value to select. Returning the one single allowed value.')
        return array_to_change
    allowed_indices = np.where((arrays != array_to_change).any(axis=1))[0]
    new_array = arrays[np.random.choice(allowed_indices)]
    return new_array


def switch_categorical_features(x, categorical_groups, categories):
    for group_features, group_categories in zip(categorical_groups, categories):
        switch_val = lambda val: any_other_array(val, group_categories)
        x[:, group_features] = np.apply_along_axis(switch_val, 1, x[:, group_features])
    return x


def switch_categorical_features_shift(x, y, categorical_groups, delta=0.5, feat_delta=1.0):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)

    # compute the categories on the whole sample, otherwise might miss some values
    # consider to put as input
    n_categorical = len(categorical_groups)
    group_indices = np.random.choice(n_categorical, ceil(n_categorical * feat_delta), replace=False)
    feat_indices = []
    categories = []
    selected_categorical_groups = [categorical_groups[i] for i in group_indices]
    for group_features in selected_categorical_groups:
        if feat_indices is not None:
            feat_indices = feat_indices.extend(group_features)
        else:
            feat_indices = group_features.copy()
        unique_rows = np.unique(x[:, np.array(group_features)], axis=0)
        categories.append(unique_rows)

    x_mod = x[indices, :]

    x_mod = switch_categorical_features(x_mod, selected_categorical_groups, categories)

    x[indices, :] = x_mod
    return x, y, indices, feat_indices


def constant_categorical(x, categorical_groups, categories):
    for group_features, group_categories in zip(categorical_groups, categories):
        allowed_indices = list(range(len(group_categories)))
        random_constant_val = group_categories[np.random.choice(allowed_indices)]
        x[:, group_features] = random_constant_val
    return x


def constant_categorical_shift(x, y, categorical_groups, delta=0.5, feat_delta=1.0):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
    x_mod = x[indices, :]

    # compute the categories on the whole sample, otherwise might miss some values
    # consider to put as input
    n_categorical = len(categorical_groups)
    group_indices = np.random.choice(n_categorical, ceil(n_categorical * feat_delta), replace=False)
    feat_indices = None
    categories = []
    selected_categorical_groups = [categorical_groups[i] for i in group_indices]
    for group_features in selected_categorical_groups:
        if feat_indices is not None:
            feat_indices.extend(group_features)
        else:
            feat_indices = group_features.copy()
        unique_rows = np.unique(x[:, np.array(group_features)], axis=0)
        categories.append(unique_rows)

    x_mod = constant_categorical(x_mod, selected_categorical_groups, categories)

    x[indices, :] = x_mod
    return x, y, indices, feat_indices


# Undersample by fraction selecting samples close to the minority class (NearMiss3 heuristics)
def under_sampling_shift(x, y, delta=0.5):
    labels = np.unique(y)
    y_counts = Counter(np.squeeze(y))
    sampling_strategy = dict()
    for label in labels:
        sampling_strategy[label] = int(delta * y_counts[label])

    # version 3 subsamples respecting more the initial data structure,
    # but it gives less control on the n of final samples (initial resampling phase)
    # thus let use version 2 as the default
    nm1 = NearMiss(version=2, sampling_strategy=sampling_strategy)
    x_resampled, y_resampled = nm1.fit_resample(x, y)
    return x_resampled, y_resampled


# Replace a fraction of samples with samples interpolated from the remaining part
def over_sampling_shift(x, y, delta=0.5, mode='smote', n_neighbors=5):

    assert(mode in ['smote', 'adasyn'])

    y_counts = Counter(np.squeeze(y))

    x_resampled, y_resampled = under_sampling_shift(x, y, delta=delta)

    n_min_samples = np.min(list(Counter(y_resampled).values()))
    n_neighbors = min(n_neighbors, n_min_samples - 1)

    if mode == 'smote':
        x_resampled, y_resampled = SMOTE(
            sampling_strategy=y_counts, k_neighbors=n_neighbors).fit_resample(x_resampled, y_resampled)
    elif mode == 'adasyn':
        x_resampled, y_resampled = ADASYN(
            sampling_strategy=y_counts, n_neighbors=n_neighbors).fit_resample(x_resampled, y_resampled)

    return x_resampled, y_resampled


# non targeted black box adversarial attacks
def adversarial_attack_shift(x, y, delta=1.0, model=RandomForestClassifier(), attack_type='zoo',
                             numerical_features=None, feat_delta=1.0):
    # in this case delta is the portion of half the data on which to generate attacks
    # because the first half as a minimum has to be used to train a model against which generate the attacks
    assert (attack_type in ['zoo', 'boundary', 'hop-skip-jump'])

    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y))
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(0.5 * delta))

    if numerical_features is not None:

        n_numerical = len(numerical_features)
        feat_indices = np.random.choice(n_numerical, ceil(n_numerical * feat_delta), replace=False)
        feat_indices = np.array(numerical_features)[feat_indices]

    else:

        feat_indices = np.random.choice(x.shape[1], ceil(x.shape[1] * feat_delta), replace=False)

    other_features = list(set(range(x.shape[1])) - set(feat_indices))

    x_train_other = x_train[:, other_features]
    x_train_numerical = x_train[:, feat_indices]
    x_test_other = x_test[:, other_features]
    x_test_numerical = x_test[:, feat_indices]

    classifier = SklearnClassifier(model=model, clip_values=(0, np.max(x_train_numerical)))

    # Train the ART classifier

    classifier.fit(x_train_numerical, y_train)

    # Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test_numerical)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Generate adversarial test examples
    if attack_type == 'zoo':
        attack = ZooAttack(
            classifier=classifier,
            confidence=0.0,
            targeted=False,
            learning_rate=1e-1,
            max_iter=10,
            binary_search_steps=10,
            initial_const=1e-3,
            abort_early=True,
            use_resize=False,
            use_importance=False,
            nb_parallel=x_test_numerical.shape[1],
            batch_size=1,
            variable_h=0.01,
        )
    elif attack_type == 'boundary':
        attack = BoundaryAttack(classifier, targeted=False, epsilon=0.02, max_iter=20, num_trial=10)
    elif attack_type == 'hop-skip-jump':
        attack = HopSkipJump(classifier,
                             targeted=False,
                             norm=2,
                             max_iter=20,
                             max_eval=10,
                             init_eval=9,
                             init_size=10)

    x_adv = attack.generate(x=x_test_numerical, y=y_test)

    # Evaluate the ART classifier on adversarial test examples

    predictions_adv = classifier.predict(x_adv)
    accuracy = np.sum(np.argmax(predictions_adv, axis=1) == y_test) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    print("Max difference: {}".format(np.max(np.abs(x_test_numerical - x_adv) / x_test_numerical)))

    x_final = np.zeros_like(x)
    x_final[:, feat_indices] = np.vstack([x_train_numerical, x_adv])
    x_final[:, other_features] = np.vstack([x_train_other, x_test_other])

    y_final = np.concatenate([y_train, y_test], axis=0)
    y_final = le.inverse_transform(y_final)

    adv_indices = list(range(len(y_train), len(y)))

    return x_final, y_final, adv_indices, feat_indices


def swap_random_features(x, corruption_probability=0.2):
    p = [corruption_probability, 1 - corruption_probability]
    (n_samples, n_features) = x.shape
    corrupt_decisions = np.random.choice(['to_corrupt', 'ok'], size=n_samples, p=p)
    corrupt_indices = np.where(corrupt_decisions == 'to_corrupt')[0]
    features_to_switch = np.random.choice(range(n_features), size=2, replace=False)
    tmp = x[corrupt_indices, features_to_switch[0]]
    x[corrupt_indices, features_to_switch[0]] = x[corrupt_indices, features_to_switch[1]]
    x[corrupt_indices, features_to_switch[1]] = tmp

    return x





