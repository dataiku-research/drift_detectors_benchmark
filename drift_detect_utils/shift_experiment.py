# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os

import itertools
import copy


def get_shift_groups(path):
    prefixes = get_folders(path)

    shift_groups = dict()
    shift_groups['covariate_gn_medium'] = []
    shift_groups['covariate_gn_small'] = []
    shift_groups['covariate_resampling'] = []
    shift_groups['covariate_constant'] = []
    shift_groups['covariate_adversarial'] = []
    shift_groups['covariate_switch'] = []
    shift_groups['prior_shift'] = []
    shift_groups['concept_shift'] = []
    shift_groups['no_shift'] = []

    for prefix in prefixes:
        if 'medium_gn_shift' in prefix:
            shift_groups['covariate_gn_medium'].append(prefix)
        elif 'small_gn_shift' in prefix:
            shift_groups['covariate_gn_small'].append(prefix)
        elif 'sample' in prefix:
            shift_groups['covariate_resampling'].append(prefix)
        elif 'adversarial' in prefix:
            shift_groups['covariate_adversarial'].append(prefix)
        elif 'switch_categorical_features' in prefix:
            shift_groups['covariate_switch'].append(prefix)
        elif 'label_switch' in prefix:
            shift_groups['concept_shift'].append(prefix)
        elif 'constant' in prefix:
            shift_groups['covariate_constant'].append(prefix)
        elif 'ko' in prefix or 'oo' in prefix or 'rebalance' in prefix:
            shift_groups['prior_shift'].append(prefix)
        elif 'no_shift' in prefix:
            shift_groups['no_shift'].append(prefix)
        else:
            print("Warning: no group for prefix %s" % prefix)

    return shift_groups


def get_prefix(name):
    parts = name.split('_')
    alpha_parts = [p for i,p in enumerate(parts) if not p.replace('.','').isdigit()]
    return '_'.join(alpha_parts)


def get_folders(path):
    list_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return list_folders


def get_results(path):

    dcl_accs = np.loadtxt("%s/dcl_accs.csv" % path, delimiter=",")
    sizes = list(dcl_accs[0])

    accs = np.loadtxt("%s/accs.csv" % path, delimiter=",")
    rand_run_p_vals = np.load("%s/rand_run_p_vals.npy" % path)

    return sizes, accs, rand_run_p_vals


class ShiftExperiment(object):
    def __init__(self, shift_type, shift_name, sizes, p_vals, accs, dr_techniques):
        self.shift_type = shift_type
        self.shift_name = shift_name
        self.sizes = np.array(sizes)
        self.p_vals = p_vals
        self.accs = accs
        self.dr_techniques = dr_techniques

        self.n_runs = self.p_vals.shape[2]
        self.n_dr = self.p_vals.shape[1]
        self.n_sizes = self.p_vals.shape[0]

        assert (self.n_dr == len(dr_techniques))

        mean_val_accs = accs[0]
        std_val_accs = accs[1]
        mean_te_accs = accs[2]
        std_te_accs = accs[3]

        n_sizes = len(sizes)
        self.accuracy_drop = np.zeros((n_sizes, 1))
        self.accuracy_drop_std = np.zeros((n_sizes, 1))
        for i in range(n_sizes):
            self.accuracy_drop[i] = mean_te_accs[i] - mean_val_accs[i]
            self.accuracy_drop_std[i] = std_val_accs[i] + std_te_accs[i]

        sorted_indices = np.argsort(self.sizes)
        self.sizes = self.sizes[sorted_indices]
        self.p_vals = self.p_vals[sorted_indices, :, :]
        self.accuracy_drop = self.accuracy_drop[sorted_indices, :]
        self.accuracy_drop_std = self.accuracy_drop_std[sorted_indices, :]

        self.sign_levels = 0.05 * np.ones((self.n_sizes, self.n_dr))

    def get_accuracy_drop(self):
        return self.accuracy_drop

    def get_min_size_to_detect(self, dr_idx, sign_level=0.05, adaptive=True, max_score=5):
        # print(self.p_vals[:, dr_idx, :])
        mean_p_vals = np.mean(self.p_vals[:, dr_idx, :], axis=1)
        if adaptive:
            drift_detected = mean_p_vals < self.sign_levels[:, dr_idx]
        else:
            drift_detected = mean_p_vals < sign_level
        detected_at_size = np.where(drift_detected == True)[0]
        if detected_at_size.size != 0 and mean_p_vals[detected_at_size[0]] != -1:
            min_size_to_detect = max(max_score - detected_at_size[0], 0)
        else:
            min_size_to_detect = 0

        return min_size_to_detect

    def get_mean_detection_accuracy(self, dr_idx, sign_level=0.05, adaptive=True):
        # print(self.p_vals[:, dr_idx, :])
        if adaptive:
            new_sign_levels = self.sign_levels[:, dr_idx]
            drift_detected = self.p_vals[:, dr_idx, :] < np.repeat(new_sign_levels[..., np.newaxis], self.n_runs,
                                                                   axis=2)
        else:
            drift_detected = self.p_vals[:, dr_idx, :] < sign_level
        mean_acc = float(np.count_nonzero(drift_detected)) / drift_detected.size
        return mean_acc

    def get_score_for_ranking(self, dr_idx, sign_level=0.05, rank_by='min_size', adaptive=True):
        assert (rank_by in ['min_size', 'mean_accuracy'])
        if rank_by == 'min_size':
            return self.get_min_size_to_detect(dr_idx, sign_level, adaptive)
        elif rank_by == 'mean_accuracy':
            return self.get_mean_detection_accuracy(dr_idx, sign_level, adaptive)

    def set_sign_levels(self, sign_levels):
        assert (self.sign_levels.shape == sign_levels.shape)
        self.sign_levels = sign_levels

    def get_sign_levels(self, dr_idx):
        sign_levels = np.zeros((self.n_sizes,))
        for s in range(self.n_sizes):
            # print(self.p_vals[s, dr_idx, :])
            quantile5perc = np.quantile(self.p_vals[s, dr_idx, :], q=0.05)
            # print(quantile5perc)
            sign_levels[s] = quantile5perc
        return sign_levels


def build_result_df_from_path(dataset_name, path, dr_techniques, results, append_col=0, sign_levels=None,
                              append_adapt=6):
    print("Building results for %s." % dataset_name)
    clf_names = results.columns

    n_sizes = len(list(sign_levels.values())[0])

    n_dr = len(dr_techniques)

    sign_levels_dr = 0.05 * np.ones((n_sizes, n_dr))
    for dr_idx, dr in enumerate(dr_techniques):
        if dr in sign_levels:
            sign_levels_dr[:, dr_idx] = sign_levels[dr]

    shift_groups = get_shift_groups(path)
    all_experiments = []
    for group in shift_groups:
        # print(group.upper())
        for shift in shift_groups[group]:
            # print(shift)
            shift_name = '_'.join([dataset_name, shift])
            try:
                shift_path = os.path.join(path, shift)
                sizes, accs, p_vals = get_results(path=shift_path)

                exp = ShiftExperiment(group, shift_name, sizes, p_vals, accs, dr_techniques)
                exp.set_sign_levels(sign_levels_dr)
                all_experiments.append(exp)

                for clf_idx, dr in enumerate(dr_techniques):
                    results.at[shift_name, clf_names[clf_idx + append_col]] = exp.get_score_for_ranking(clf_idx,
                                                                                                        adaptive=False)
                    results.at[shift_name, clf_names[clf_idx + append_col + append_adapt]] = exp.get_score_for_ranking(
                        clf_idx, adaptive=True)
            except Exception as e:
                print('skipping ' + dataset_name + ' ' + dr_techniques[0] + ' ' + shift)
                print(e)
                continue
    return results, all_experiments


def build_sign_levels(dataset_name, path, dr_techniques):
    print("Building adaptive significance levels for %s." % dataset_name)

    shift = 'no_shift'
    shift_name = '_'.join([dataset_name, shift])

    shift_path = os.path.join(path, shift)
    sizes, accs, p_vals = get_results(path=shift_path)

    exp = ShiftExperiment('no_shift', shift_name, sizes, p_vals, accs, dr_techniques)
    sign_levels = dict()
    for dr_idx, dr in enumerate(dr_techniques):
        sign_levels[dr] = exp.get_sign_levels(dr_idx)

    return sign_levels, exp


def build_result_df(dataset_name, path_rf, path_no, dr_techniques, adapt=True, dr_techniques_no=None):
    shift_adapt = 0
    clf_names = copy.deepcopy(dr_techniques)
    if adapt:
        shift_adapt = len(dr_techniques)
        clf_names += [name + ' (adapt)' for name in clf_names]

    shift_groups = get_shift_groups(path_rf)
    data_names = list(itertools.chain.from_iterable(shift_groups.values()))
    data_names = ['_'.join([dataset_name, d]) for d in data_names]
    results = pd.DataFrame(index=data_names, columns=clf_names, dtype=float)

    if dr_techniques_no is None:
        dr_techniques_no = dr_techniques
    sign_levels, no_experiment = build_sign_levels(dataset_name, path_no, dr_techniques_no)

    results, all_rf_experiments = build_result_df_from_path(dataset_name, path_rf, dr_techniques, results,
                                                            append_col=0, sign_levels=sign_levels,
                                                            append_adapt=shift_adapt)

    return results, all_rf_experiments, no_experiment
