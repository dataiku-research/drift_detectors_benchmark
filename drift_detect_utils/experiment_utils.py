from drift_detect_utils.shift_applicator import apply_shift
from failing_loudly.shift_detector import ShiftDetector
from failing_loudly.shared_utils import DimensionalityReduction, OnedimensionalTest, TestDimensionality
from failing_loudly.shift_locator import DifferenceClassifier, ShiftLocator
from drift_detect_utils.dataset_utils import get_rand_dataset_split
import numpy as np
import os
import pickle
import shutil
import copy
import fnmatch


def rand_runs_drift_detection_quality(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                                      qualities, out_path, sample,
                                      random_runs=5, sign_level=0.05):
    od_tests = [OnedimensionalTest.KS.value]
    md_tests = []
    test_types = [td.value for td in TestDimensionality]

    # Define DR methods.
    dr_techniques = [DimensionalityReduction.BBSDs.value]
    dr_techniques_plot = dr_techniques.copy()

    shift_path = os.path.join(out_path, shift)
    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    datset = df_train_name

    red_models = [None] * len(DimensionalityReduction)

    # Stores p-values
    rand_run_p_vals = np.ones((len(qualities), len(dr_techniques_plot), random_runs)) * (-1)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(qualities))) * (-1)
    te_accs = np.ones((random_runs, len(qualities))) * (-1)

    # Average over a few random runs to quantify robustness.

    for rand_run in range(random_runs):

        print("Random run %s" % rand_run)

        np.random.seed(rand_run)

        print("Loading data...")
        # Load data.

        dataframes, processed_data, numerical_features, categorical_groups = get_rand_dataset_split(df_train_name,
                                                                                                    df_valid_name,
                                                                                                    df_test_name,
                                                                                                    target, max_num_row,
                                                                                                    rand_run, out_path)

        [(X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes,
         le, feature_names] = processed_data

        X_te_orig_copy = copy.deepcopy(X_te_orig)

        (X_te_orig, y_te_orig), shifted_indices, shifted_feat_indices = apply_shift(
            X_te_orig, y_te_orig, shift,
            numerical_features=numerical_features,
            categorical_groups=categorical_groups)

        print("%%%%%%%%%%%%%%%%% Summary - Shift")
        print(X_te_orig.shape)
        print(np.unique(y_te_orig, return_counts=True))
        if shifted_indices is not None:
            print('Len shifted: ', len(shifted_indices))
            l2 = np.linalg.norm(np.fabs(X_te_orig_copy - X_te_orig), axis=1)
            print('max L2 diff: ', np.max(l2))
            print('mean L2 diff: ', np.mean(l2))

        # permutation is needed here (for shift)
        n_total_samples = min(len(y_val_orig), len(y_te_orig))
        if n_total_samples % 2:
            n_total_samples = n_total_samples - 1
        permuted_indices = np.random.permutation(n_total_samples)

        X_te_orig = X_te_orig[permuted_indices, :]
        y_te_orig = y_te_orig[permuted_indices]

        X_val_orig = X_val_orig[permuted_indices, :]
        y_val_orig = y_val_orig[permuted_indices]

        if shifted_indices is not None:
            shifted_indices = [list(permuted_indices).index(s) for s in shifted_indices if s in permuted_indices]

        with open(f'{shift_path}/shifted_data_{rand_run}.pkl', 'wb') as f:
            pickle.dump([(X_te_orig, y_te_orig), shifted_indices, shifted_feat_indices], f)

        with open(f'{shift_path}/permuted_val_data_{rand_run}.pkl', 'wb') as f:
            pickle.dump((X_val_orig, y_val_orig), f)

        # Check detection performance for different quality of primary BBSD model.
        for qi, quality in enumerate(qualities):

            print("Quality %s" % quality)

            n_total_samples = min(len(y_val_orig), len(y_te_orig))
            if sample > n_total_samples:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("Skipping: samples %s > max size %s" % (sample, n_total_samples))
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                continue

            X_te_3 = X_te_orig[:sample, :]
            y_te_3 = y_te_orig[:sample]

            X_val_3 = X_val_orig[:sample, :]
            y_val_3 = y_val_orig[:sample]

            X_tr_3 = np.copy(X_tr_orig)
            y_tr_3 = np.copy(y_tr_orig)

            # Detect shift.
            shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models,
                                           sample, datset, quality)

            (od_decs, ind_od_decs, ind_od_p_vals), \
            (md_decs, ind_md_decs, ind_md_p_vals), \
            red_dim, red_models, val_acc, te_acc = shift_detector.detect_data_shift(X_tr_3, y_tr_3, X_val_3, y_val_3,
                                                                                    X_te_3, y_te_3, orig_dims,
                                                                                    nb_classes)

            val_accs[rand_run, qi] = val_acc
            te_accs[rand_run, qi] = te_acc

            print("Shift decision: ", ind_od_decs.flatten())
            print("Shift p-vals: ", ind_od_p_vals.flatten())

            rand_run_p_vals[qi, :, rand_run] = ind_od_p_vals.flatten()

    np.save("%s/rand_run_p_vals.npy" % (out_path), rand_run_p_vals)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    accs = np.ones((4, len(qualities))) * (-1)
    accs[0] = mean_val_accs
    accs[1] = std_val_accs
    accs[2] = mean_te_accs
    accs[3] = std_te_accs

    print("Mean Val accuracies: ", mean_val_accs.flatten())
    print("Mean Test accuracies: ", mean_te_accs.flatten())

    np.savetxt("%s/accs.csv" % out_path, accs, delimiter=",")


def rand_runs_drift_detection(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                              dr_techniques, dc_techniques,
                              samples, out_path,
                              random_runs=5, sign_level=0.05,
                              features_to_shift=None,
                              save_models=False,
                              save_dataset=False,
                              suffix=''):
    od_tests = [OnedimensionalTest.KS.value]
    md_tests = []
    test_types = [td.value for td in TestDimensionality]

    dr_techniques_plot = dr_techniques.copy()
    dr_techniques_plot.extend(dc_techniques)

    if DimensionalityReduction.Classif_RF.value in dc_techniques:
        dc_model = DifferenceClassifier.RF
    else:
        dc_model = DifferenceClassifier.FFNNDCL

    shift_path = os.path.join(out_path, shift)
    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    datset = df_train_name

    red_models = [None] * len(DimensionalityReduction)

    # Stores p-values
    rand_run_p_vals = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)
    dcl_scores = np.ones((len(samples), random_runs)) * (-1)

    # Average over a few random runs to quantify robustness.

    n_max_samples = np.max(samples)
    print(n_max_samples)

    for rand_run in range(random_runs):

        print("Random run %s" % rand_run)

        np.random.seed(rand_run)

        dataframes, processed_data, numerical_features, categorical_groups = get_rand_dataset_split(df_train_name,
                                                                                                    df_valid_name,
                                                                                                    df_test_name,
                                                                                                    target, max_num_row,
                                                                                                    rand_run, out_path,
                                                                                                    save_dataset=save_dataset)

        try:
            [(X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes,
             le, feature_names] = processed_data
        except ValueError as e:
            print(e)
            print('Do not load feature names')
            [(X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes,
             le] = processed_data

        shifted_data_filename = f'{shift_path}/shifted_data_{rand_run}.pkl'
        permuted_data_filename = f'{shift_path}/permuted_val_data_{rand_run}.pkl'

        if os.path.exists(shifted_data_filename) and os.path.exists(permuted_data_filename):
            print("Loading from file: ", shifted_data_filename)
            with open(shifted_data_filename, 'rb') as f:
                [(X_te_orig, y_te_orig), shifted_indices, shifted_feat_indices] = pickle.load(f)

            print("Loading from file: ", permuted_data_filename)
            with open(permuted_data_filename, 'rb') as f:
                (X_val_orig, y_val_orig) = pickle.load(f)

            print("%%%%%%%%%%%%%%%%% Summary - Shift")
            print(X_te_orig.shape)
            print(np.unique(y_te_orig, return_counts=True))

        else:

            X_te_orig = X_te_orig[:n_max_samples, :]
            y_te_orig = y_te_orig[:n_max_samples]

            X_val_orig = X_val_orig[:n_max_samples, :]
            y_val_orig = y_val_orig[:n_max_samples]

            X_te_orig_copy = copy.deepcopy(X_te_orig)

            (X_te_orig, y_te_orig), shifted_indices, shifted_feat_indices = apply_shift(
                X_te_orig, y_te_orig, shift,
                numerical_features=numerical_features,
                categorical_groups=categorical_groups,
                # min_size=np.max(samples[:-1]),  # resample with min size at least 1000
                features_to_shift=features_to_shift)

            print("%%%%%%%%%%%%%%%%% Summary - Shift")
            print(X_te_orig.shape)
            print(np.unique(y_te_orig, return_counts=True))
            if shifted_indices is not None:
                print('Len shifted: ', len(shifted_indices))
                l2 = np.linalg.norm(np.fabs(X_te_orig_copy - X_te_orig), axis=1)
                print('max L2 diff: ', np.max(l2))
                print('mean L2 diff: ', np.mean(l2))
            if shifted_feat_indices is not None:
                print("N feat shifted: ", len(shifted_feat_indices))

            # permutation is needed here (for shift)
            n_total_samples = min(len(y_val_orig), len(y_te_orig))
            if n_total_samples % 2:
                n_total_samples = n_total_samples - 1
            permuted_indices = np.random.permutation(n_total_samples)

            X_te_orig = X_te_orig[permuted_indices, :]
            y_te_orig = y_te_orig[permuted_indices]

            X_val_orig = X_val_orig[permuted_indices, :]
            y_val_orig = y_val_orig[permuted_indices]

            if shifted_indices is not None:
                shifted_indices = [list(permuted_indices).index(s) for s in shifted_indices if s in permuted_indices]

            if save_dataset:
                with open(shifted_data_filename, 'wb') as f:
                    pickle.dump([(X_te_orig, y_te_orig), shifted_indices, shifted_feat_indices], f)

                with open(permuted_data_filename, 'wb') as f:
                    pickle.dump((X_val_orig, y_val_orig), f)

        # Check detection performance for different numbers of samples from test.
        for si, sample in enumerate(samples):

            print("Sample %s" % sample)

            n_total_samples = min(len(y_val_orig), len(y_te_orig))
            if sample > n_total_samples:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("Skipping: samples %s > max size %s" % (sample, n_total_samples))
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                continue

            X_te_3 = X_te_orig[:sample, :]
            y_te_3 = y_te_orig[:sample]

            X_val_3 = X_val_orig[:sample, :]
            y_val_3 = y_val_orig[:sample]

            X_tr_3 = np.copy(X_tr_orig)
            y_tr_3 = np.copy(y_tr_orig)

            print("%%%%%%%%%%%%%%%%% Summary - Subsample")
            print("Val")
            print(X_val_3.shape)
            print(np.unique(y_val_3, return_counts=True))
            print("Test")
            print(X_te_3.shape)
            print(np.unique(y_te_3, return_counts=True))

            # Detect shift.
            shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models,
                                           sample, datset)

            (od_decs, ind_od_decs, ind_od_p_vals), \
            (md_decs, ind_md_decs, ind_md_p_vals), \
            red_dim, red_models, val_acc, te_acc = shift_detector.detect_data_shift(X_tr_3, y_tr_3, X_val_3, y_val_3,
                                                                                    X_te_3, y_te_3, orig_dims,
                                                                                    nb_classes)

            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            print("Val acc: ", val_acc)
            print("Test acc: ", te_acc)

            print("Shift decision: ", ind_od_decs.flatten())
            print("Shift p-vals: ", ind_od_p_vals.flatten())

            if DimensionalityReduction.Classif.value in dr_techniques_plot or \
                    DimensionalityReduction.Classif_RF.value in dr_techniques_plot:

                # Characterize shift via domain classifier.
                shift_locator = ShiftLocator(orig_dims, dc=dc_model, sign_level=sign_level)
                model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old), test_idx_mapping = \
                    shift_locator.build_model(
                    X_val_3, y_val_3,
                    X_te_3, y_te_3)

                test_indices, test_perc, dec, p_val = shift_locator.most_likely_shifted_samples(model, X_te_dcl,
                                                                                                y_te_dcl)

                shifted_test_data_filename = f'{shift_path}/shifted_te_data_{rand_run}_{sample}.pkl'

                if os.path.exists(shifted_test_data_filename):
                    print("Loading from file: ", shifted_test_data_filename)
                    with open(shifted_test_data_filename, 'rb') as f:
                        [(X_te_dcl, y_te_old), shifted_indices_test, shifted_feat_indices] = pickle.load(f)

                else:

                    shifted_indices_test = None
                    if shifted_indices is not None:
                        shifted_indices_test = [test_idx_mapping[i] for i in shifted_indices if i in test_idx_mapping]

                    if save_dataset:
                        with open(shifted_test_data_filename, 'wb') as f:
                            pickle.dump([(X_te_dcl, y_te_old), shifted_indices_test, shifted_feat_indices], f)

                if save_models:
                    with open("%s/dc_model_%d_%d.pkl" % (shift_path, rand_run, sample), 'wb') as f:
                        pickle.dump(model, f)
                    with open("%s/dc_model_score_%d_%d.pkl" % (shift_path, rand_run, sample), 'wb') as f:
                        pickle.dump(score, f)

                rand_run_p_vals[si, :, rand_run] = np.append(ind_od_p_vals.flatten(), p_val)
                dcl_scores[si, rand_run] = score
                print("Domain Classifier score: %.2f" % score)
                print("Domain Classifier p-value: %.2f " % p_val)

                if dec:
                    most_conf_test_indices = test_indices[test_perc > 0.8]

                    print('-------------------')
                    print("Len of most conf: %s" % len(most_conf_test_indices))

                    if len(most_conf_test_indices) > 0:
                        y_te_dcl_pred = shift_detector.classify_data(X_tr_3, y_tr_3, X_val_3, y_val_3,
                                                                     X_te_dcl[most_conf_test_indices],
                                                                     y_te_dcl[most_conf_test_indices],
                                                                     orig_dims, nb_classes,
                                                                     dr_technique=DimensionalityReduction.BBSDh_RF)

                        dcl_class_acc = np.sum(np.equal(np.squeeze(y_te_dcl_pred), y_te_old[most_conf_test_indices])
                                               .astype(int)) / len(y_te_dcl_pred)
                        dcl_accs[si, rand_run] = dcl_class_acc
                        print("dcl_class_acc: ", dcl_class_acc)
                    print('-------------------')
            else:
                rand_run_p_vals[si, :, rand_run] = ind_od_p_vals.flatten()

        if rand_run == 0 and save_models:
            with open("%s/pr_model.pkl" % shift_path, 'wb') as f:
                pickle.dump(red_models[0], f)

    rand_run_p_vals_filename = f'{shift_path}/rand_run_p_vals{suffix}.npy'
    val_accs_filename = f'{shift_path}/val_accs{suffix}.npy'
    te_accs_filename = f'{shift_path}/te_accs{suffix}.npy'
    accs_filename = f'{shift_path}/accs{suffix}.csv'
    dcl_accs_filename = f'{shift_path}/dcl_accs{suffix}.csv'

    np.save(rand_run_p_vals_filename, rand_run_p_vals)
    np.save(val_accs_filename, val_accs)
    np.save(te_accs_filename, te_accs)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    mean_dcl_accs = []
    std_dcl_accs = []
    for si, sample in enumerate(samples):
        cur_dcl_accs = dcl_accs[si, :]
        valid_indices = np.where(cur_dcl_accs != -1)[0]
        if valid_indices.any():
            mean_dcl_accs.append(np.mean(cur_dcl_accs[valid_indices]))
            std_dcl_accs.append(np.std(cur_dcl_accs[valid_indices]))
        else:
            mean_dcl_accs.append(-1)
            std_dcl_accs.append(-1)

    mean_dcl_accs = np.array(mean_dcl_accs)
    std_dcl_accs = np.array(std_dcl_accs)
    smpl_array = np.array(samples)

    mean_dcl_scores = np.mean(dcl_scores, axis=1)

    print("mean_dcl_scores: ", mean_dcl_scores)
    print("mean_dcl_accs: ", mean_dcl_accs)
    print("std_dcl_accs: ", std_dcl_accs)
    print("smpl_array: ", smpl_array)

    accs = np.ones((4, len(samples))) * (-1)
    accs[0] = mean_val_accs
    accs[1] = std_val_accs
    accs[2] = mean_te_accs
    accs[3] = std_te_accs

    dcl_accs = np.ones((4, len(smpl_array))) * (-1)
    dcl_accs[0] = smpl_array
    dcl_accs[1] = mean_dcl_accs
    dcl_accs[2] = std_dcl_accs
    dcl_accs[3] = mean_dcl_scores

    np.savetxt(accs_filename, accs, delimiter=",")
    np.savetxt(dcl_accs_filename, dcl_accs, delimiter=",")


def delete_existing_red_models(saved_models_folder='./saved_models'):
    # delete existing reduction models
    if os.path.exists(saved_models_folder):
        for filename in os.listdir(saved_models_folder):
            file_path = os.path.join(saved_models_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def delete_existing_datasets(path):
    # delete existing datasets
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                if fnmatch.fnmatch(filename, "train_dataframe.pkl") or \
                        fnmatch.fnmatch(filename, "valid_test_dataframes_*.pkl") or \
                        fnmatch.fnmatch(filename, "processed_data_*.pkl"):
                    try:
                        print("Remove ", file_path)
                        os.remove(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))
            elif os.path.isdir(file_path):
                for sub_filename in os.listdir(file_path):
                    sub_file_path = os.path.join(file_path, sub_filename)
                    if os.path.isfile(sub_file_path):
                        if fnmatch.fnmatch(sub_filename, "shifted_data_*.pkl") or \
                                fnmatch.fnmatch(sub_filename, "permuted_val_data_*.pkl") or \
                                fnmatch.fnmatch(sub_filename, "shifted_te_data_*.pkl") or \
                                fnmatch.fnmatch(sub_filename, "shifted_tr_data_*.pkl"):
                            try:
                                print("Remove ", sub_file_path)
                                os.remove(sub_file_path)
                            except Exception as e:
                                print('Failed to delete %s. Reason: %s' % (sub_file_path, e))
