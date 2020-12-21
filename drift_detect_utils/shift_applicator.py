# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from .perturbations import *


# -------------------------------------------------
# SHIFT APPLICATOR
# -------------------------------------------------

noise_levels = {
                  'large': 100.0,
                  'medium': 10.0,
                  'small': 1.0
}


def apply_shift(X_te_orig, y_te_orig, shift,
                features_to_shift=None, numerical_features=None, categorical_groups=None,
                min_size=5000):
    X_te_1 = None
    y_te_1 = None
    indices = None
    feat_indices = None

    feat_delta = 0.
    if not (features_to_shift is None):
        numerical_features = list(set(numerical_features) & set(features_to_shift))
        categorical_groups = [g for g in categorical_groups if set(g) & set(features_to_shift)]
        feat_delta = 1.0

    print(shift.replace('_', ' ').title())

    if shift == 'no_shift':
        X_te_1 = X_te_orig.copy()
        y_te_1 = y_te_orig.copy()
    elif 'ko_shift' in shift:
        params = shift.split('_')
        delta = float(params[2])
        if len(params) == 4:
            cl = int(params[3])
        else:
            cl = y_te_orig[0]
        X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, cl, delta)
    elif 'oo_shift' in shift:
        params = shift.split('_')
        if len(params) == 3:
            keep_cl = int(params[2])
        else:
            keep_cl = y_te_orig[0]
        X_te_1, y_te_1 = only_one_shift(X_te_orig, y_te_orig, keep_cl)
    elif 'rebalance_shift' in shift:
        params = shift.split('_')
        priors = [float(p) for p in params[2:]]
        X_te_1, y_te_1 = rebalance_shift(X_te_orig, y_te_orig, priors)
    elif 'gn_shift' in shift:
        params = shift.split('_')
        noise_key = params[0]
        noise = noise_levels[noise_key]
        delta = float(params[3])
        feat_delta = max(feat_delta, float(params[4]))
        X_te_1, _, indices, feat_indices = gaussian_noise_shift(X_te_orig, y_te_orig,
                                                                noise, delta, feat_delta,
                                                                numerical_features=numerical_features)
        y_te_1 = y_te_orig.copy()
    elif 'switch_categorical_features_shift' in shift:
        params = shift.split('_')
        delta = float(params[4])
        feat_delta = max(feat_delta, float(params[5]))
        X_te_1, y_te_1, indices, feat_indices = switch_categorical_features_shift(X_te_orig, y_te_orig,
                                                                                  categorical_groups,
                                                                                  delta=delta,
                                                                                  feat_delta=feat_delta)
    elif 'subsample_joint_shift' in shift:
        params = shift.split('_')
        gamma = 0.8
        if len(params) == 4:
            gamma = float(params[3])
        X_te_1, y_te_1, feat_indices = subsample_joint_shift(X_te_orig, y_te_orig,
                                                             gamma=gamma,
                                                             shift_features=numerical_features)
    elif 'subsample_feature_shift' in shift:
        params = shift.split('_')
        feat_delta = max(feat_delta, float(params[3]))
        X_te_1, y_te_1, feat_indices = subsample_feature_shift(X_te_orig, y_te_orig,
                                                               feat_delta=feat_delta,
                                                               numerical_features=numerical_features,
                                                               min_size=min_size)
    elif 'subsample_categorical_feature_shift' in shift:
        X_te_1, y_te_1 = subsample_all_categorical_feature_shift(X_te_orig, y_te_orig,
                                                                 categorical_groups=categorical_groups,
                                                                 min_size=min_size)

    elif 'label_switch_shift' in shift:
        params = shift.split('_')
        perc_max_changes = float(params[-1])
        X_te_1, y_te_1, indices = random_class_subset_shift(X_te_orig, y_te_orig, perc_max_changes)
    elif 'under_sample_shift' in shift:
        params = shift.split('_')
        delta = float(params[-1])
        X_te_1, y_te_1 = under_sampling_shift(X_te_orig, y_te_orig, delta)
    elif 'over_sample_shift' in shift:
        params = shift.split('_')
        delta = float(params[-1])
        X_te_1, y_te_1 = over_sampling_shift(X_te_orig, y_te_orig, delta)
    elif 'adversarial_attack_shift' in shift:
        params = shift.split('_')
        delta = float(params[4])
        if 'boundary' in shift:
            attack_type = 'boundary'
        else:
            attack_type = 'zoo'
        if len(params) == 6:
            feat_delta = max(feat_delta, float(params[5]))
        else:
            feat_delta = 1.0
        X_te_1, y_te_1, indices, feat_indices = adversarial_attack_shift(X_te_orig, y_te_orig,
                                                                         delta, attack_type=attack_type,
                                                                         numerical_features=numerical_features,
                                                                         feat_delta=feat_delta)

    elif 'constant_feature_shift' in shift:

        params = shift.split('_')

        delta = float(params[3])

        feat_delta = max(feat_delta, float(params[4]))

        X_te_1, y_te_1, indices, feat_indices = constant_all_feature_shift(X_te_orig, y_te_orig,
                                                                           delta=delta, feat_delta=feat_delta,
                                                                           numerical_features=numerical_features,
                                                                           categorical_groups=categorical_groups)

    return (X_te_1, y_te_1), indices, feat_indices
