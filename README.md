# Drift Detectors Benchmark

Software to run shift detection experiments and benchmark state-of-the-art drift detectors on structured datasets.

## Requirements

This software re-uses and modifies some parts of the 
[failing_loudly](https://github.com/steverab/failing-loudly/tree/024dda322287de1ddb4b2849957b27936df681dd) 
repository, which is thus not included in the requirements. The modifications are meant to adapt the shift detectors in 
the failing_loudly repository to work with structured datasets. 

```bash
sklearn
scipy==1.4.1
pandas
imbalanced-learn==0.6.2
adversarial-robustness-toolbox==1.2.0
torch==1.4.0
git+https://github.com/josipd/torch-two-sample.git#egg=torch-two-sample
keras==2.2.4
keras-resnet==0.1.0
tensorflow==1.13.1
```

## Usage

This software assumes that the dataset is provided unprocessed and separated in a three-way splits (train, validation and test) 
as three different .csv files, which will be loaded as pandas dataframe (see the example of [Click_prediction_small](https://www.openml.org/d/1226) 
dataset below). The test dataset will be corrupted by synthetic shifts and compared to the unperturbed validation dataset.
The target column for the dataset must also be provided in the `target` variable.

You can run the `example.py` script for a full view of the experiments on the Click_prediction_small dataset.
 
### Run drift detection on multiple types of synthetic shifts

```python
from drift_detect_utils.experiment_utils import rand_runs_drift_detection
from failing_loudly.shared_utils import DimensionalityReduction

# Read inputs
dataset_name = 'Click_prediction_small'
df_train_name = dataset_name + '_train.csv'
df_valid_name = dataset_name + '_valid.csv'
df_test_name = dataset_name + '_test.csv'
target = 'click'
max_num_row = 10000

# Write outputs
out_path = dataset_name + '_drift_rf'

# Define DR methods.
dr_techniques = [DimensionalityReduction.BBSDs_RF.value, DimensionalityReduction.BBSDh_RF.value, 
                 DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, 
                 DimensionalityReduction.SRP.value]
dc_techniques = [DimensionalityReduction.Classif_RF.value]

# Define shift types.

shifts = ['no_shift']

shifts += ['ko_shift_0.4_0', 'ko_shift_0.1_0']
shifts += ['oo_shift_0']
shifts += ['medium_gn_shift_1.0_1.0', 'medium_gn_shift_0.5_1.0', 'medium_gn_shift_0.5_0.5', 'medium_gn_shift_1.0_0.5',
          'small_gn_shift_1.0_1.0', 'small_gn_shift_0.5_1.0', 'small_gn_shift_0.5_0.5', 'small_gn_shift_1.0_0.5']
shifts += ['subsample_joint_shift', 'subsample_feature_shift_1.0']
shifts += ['under_sample_shift_0.5', 'over_sample_shift_0.5']
shifts += ['switch_categorical_features_shift_0.5_1.0', 'subsample_categorical_feature_shift']
shifts += ['adversarial_attack_shift_zoo_0.5', 'adversarial_attack_shift_boundary_0.5']
shifts += ['adversarial_attack_shift_zoo_1.0', 'adversarial_attack_shift_boundary_1.0']

samples = [10, 100, 500, 1000, 2000]
n_runs = 5

for shift in shifts:

    print('Running Drift Experiments on %s for %s' % (df_train_name, shift))
        
    rand_runs_drift_detection(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                            dr_techniques, dc_techniques, samples, out_path, random_runs=n_runs, sign_level=0.05)       

```

### Run drift detection on multiple splits of unperturbed datasets (to compute a dataset-adaptive significance level)

```python
from drift_detect_utils.experiment_utils import rand_runs_drift_detection
from failing_loudly.shared_utils import DimensionalityReduction

# Read inputs as above
# ...
# Define DR methods as above
#...
# Write outputs
out_folder = dataset_name + '_drift_no'

shift = 'no_shift'
samples = [10, 100, 500, 1000, 2000]
n_runs = 100
print('Running Drift Experiments on %s for %s' % (df_train_name, shift))
        
rand_runs_drift_detection(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                          dr_techniques, dc_techniques, samples, out_path, random_runs=n_runs, sign_level=0.05)       

```

### Run drift detection on multiple types of synthetic shifts, while corrupting the quality of the primary model

```python
from drift_detect_utils.experiment_utils import rand_runs_drift_detection_quality

# Read inputs as above
# ...
# Define DR methods.
# This experiment is only defined for DimensionalityReduction.BBSDs_RF.value

# Write outputs
out_path = dataset_name + '_drift_quality'

# Define shift types.

shifts = ['no_shift']
shifts += ['ko_shift_0.4_0', 'ko_shift_0.1_0']
shifts += ['oo_shift_0']

qualities = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
sample = 1000
n_runs = 5

for shift in shifts:

    print('Running Drift Experiments on %s for %s' % (df_train_name, shift))
        
    rand_runs_drift_detection_quality(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                                      qualities, out_path, sample, random_runs=n_runs, sign_level=0.05)       

```

### Read the Results

The results are saved as several numpy files (as in failing_loudly), which can be loaded via `build_result_df` as shown below. 
The output `results` is a pandas dataframe containing one line per shift and as many columns as the different shift detectors in the experiment. 
Each value in the dataframe is the *efficiency score* computed for each detector and shift: a number between 0 and the number of sizes tested in `samples`.
The highest the efficiency score, the fewest the samples needed to detect the drift. The output  `all_rf_experiments` and `no_experiment` 
are lists of `drift_detect_utils.shift_experiment.ShiftExperiment` objects, containing all information regarding the individual experiments, 
for instance the p-values in `ShiftExperiment.p_vals` of size [`n_sizes`, `n_detectors`, `n_runs`], or the mean accuracy drops in `ShiftExperiment.accuracy_drop`.
The efficiency score can be retrieved through `ShiftExperiment.get_score_for_ranking`.

```python
from drift_detect_utils.shift_experiment import build_result_df

dataset_name = 'Click_prediction_small'
out_path_rf = dataset_name + '_drift_rf' 
out_path_no = dataset_name + '_drift_no' 

dr_techniques = ['BBSDs', 'BBSDh', 'Test_X', 'Test_PCA', 'Test_SRP', 'DC']
results, all_rf_experiments, no_experiment = build_result_df(dataset_name, out_path_rf, out_path_no,
                                                             dr_techniques, adapt=True)

```

For the experiments of the Black-Box-Shift-Detector with corruption of the primary model quality, the results can be 
loaded through `drift_detect_utils.shift_experiment.load_all_quality_results` which returns a pandas dataframe with one
row per shift type, quality and run, containing the p-values from the Black-Box-Shift-Detector KS-test. 

```python
from drift_detect_utils.shift_experiment import load_all_quality_results

dataset_name = 'Click_prediction_small'
out_path_quality = dataset_name + '_drift_quality' 

results = load_all_quality_results(out_path_quality)

```

## License
