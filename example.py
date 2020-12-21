from drift_detect_utils.experiment_utils import rand_runs_drift_detection, rand_runs_drift_detection_quality
from drift_detect_utils.shift_experiment import build_result_df, load_all_quality_results
from failing_loudly.shared_utils import DimensionalityReduction

# Read inputs
dataset_name = 'Click_prediction_small'
df_train_name = dataset_name + '_train.csv'
df_valid_name = dataset_name + '_valid.csv'
df_test_name = dataset_name + '_test.csv'
target = 'click'
max_num_row = 10000

# Write outputs
out_path_rf = dataset_name + '_drift_rf'
out_path_no = dataset_name + '_drift_no'
out_path_quality = dataset_name + '_drift_quality'

# Define DR methods.
dr_techniques = [DimensionalityReduction.BBSDs_RF.value, DimensionalityReduction.BBSDh_RF.value,
                 DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value,
                 DimensionalityReduction.SRP.value]
dc_techniques = [DimensionalityReduction.Classif_RF.value]

# Define shift types.

shifts = ['no_shift']
shifts += ['ko_shift_0.4_0', 'ko_shift_0.1_0']

samples = [10, 100, 500, 1000, 2000]
n_runs = 5

for shift in shifts:
    print('Running Drift Experiments on %s for %s' % (df_train_name, shift))

    rand_runs_drift_detection(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                              dr_techniques, dc_techniques, samples, out_path_rf, random_runs=n_runs, sign_level=0.05)


shift = 'no_shift'
n_runs = 10

print('Running Drift Experiments on %s for %s' % (df_train_name, shift))

rand_runs_drift_detection(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                          dr_techniques, dc_techniques, samples, out_path_no, random_runs=n_runs, sign_level=0.05)


dr_techniques = ['BBSDs', 'BBSDh', 'Test_X', 'Test_PCA', 'Test_SRP', 'DC']
results, all_rf_experiments, no_experiment = build_result_df(dataset_name, out_path_rf, out_path_no,
                                                             dr_techniques, adapt=True)

print(results)

qualities = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
sample = 1000
n_runs = 5

for shift in shifts:
    print('Running Drift Experiments on %s for %s' % (df_train_name, shift))

    rand_runs_drift_detection_quality(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                                      qualities, out_path_quality, sample, random_runs=n_runs, sign_level=0.05)

results_quality = load_all_quality_results(out_path_quality)

print(results_quality)



