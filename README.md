# Drift Detectors Benchmark

Software to run shift detection experiments and benchmark state-of-the-art drift detectors.

## Requirements

```bash
sklearn
scipy
imblearn
adversarial-robustness-toolbox
```

## Usage

```python
from drift_dac_utils.experiment_utils import rand_runs_drift_detection
from failing_loudly.shared_utils import DimensionalityReduction

# Read inputs
dataset_name = 'general_Click_prediction_small'
df_train_name = dataset_name + '_train'
df_valid_name = dataset_name + '_valid'
df_test_name = dataset_name + '_test'
target = 'click'
max_num_row = 10000

# Write outputs
out_folder = dataset_name + '_drift_rf'

# Define DR methods.
dr_techniques = [DimensionalityReduction.BBSDs_RF.value]
dc_techniques = [DimensionalityReduction.Classif_RF.value]

# Define shift types.

shift = 'small_gn_shift_0.5_1.0'

samples = [10, 100, 500, 1000, 2000]

print('Running Drift Experiments on %s for %s' % (df_train_name, shift))
        
rand_runs_drift_detection(shift, df_train_name, df_valid_name, df_test_name, target, max_num_row,
                          dr_techniques, dc_techniques, samples, out_path,
                          random_runs=5, sign_level=0.05)

```

## License
