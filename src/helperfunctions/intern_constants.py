
from .getprojectroot import define_project_root_path
PATH_PROJECT_ROOT =define_project_root_path()


PATH_PENMANSHIEL = (PATH_PROJECT_ROOT / "src" / "Penmanshiel").resolve()

PATH_TRAINING_CFG = PATH_PROJECT_ROOT / "ae_training_configs"

PATH_FEATURE_ORDER = PATH_PENMANSHIEL / "feature_order"

PATH_FEATURE_ORDER_FILE = PATH_FEATURE_ORDER / "feature_order.json"

PATH_PROCESSED_DATA = PATH_PENMANSHIEL / "processed_data"

PATH_IMPUTED = PATH_PROCESSED_DATA / "signal_imputation"

PATH_IMPUT_MASKS = PATH_IMPUTED / "masks"

PATH_IDIO_COMP = PATH_IMPUTED / "idio_comps_and_fleet_median"

PATH_FLEETMEDIAN = PATH_IDIO_COMP / "fleet_median"

#PATH_FM_DL = PATH_FLEETMEDIAN / "fm_dataloader"

PATH_MINMAX = PATH_IMPUTED / "minmax_scaling"

PATH_PC_FILTERING = PATH_MINMAX / "cleaned_data_pc_filtering"

PATH_QUANTILE_FILTERING = PATH_MINMAX / "quantile_filtering"

PATH_POWERCURVE_MASKS = PATH_PC_FILTERING / "pc_masks"

PATH_MINMAX_SCALER = PATH_MINMAX / "minmax_lookup_table.pkl"


# Early stopping, best model
PATH_TO_BEST_MODEL_DIR = PATH_PROJECT_ROOT / "src" / "early_stopping"

BEST_MODEL = "best_model.pth"

# Directory for visual prints e.g. Learning curve, mean turbine loss per sample ..
#DO NOT CHANGE
DIRECTORY_PRINTS = "prints"
PATH_PRINTS = PATH_PROJECT_ROOT / "src" / DIRECTORY_PRINTS
#FIXED train time period
TRAIN_START  = "2018-04-05 13:50:00"
TRAIN_END = "2019-04-05 13:50:00"


# For visualization/evaluation of losses during and after training
MEAN_LOSS_PER_SAMPLE = "Mean Loss per Sample"
RE_PREFIX ="RE_"
SIGNAL_COL = "Signal"
RE_COL = "Reconstruction Error"

# Column name of Date and Time in df tables
TS_COL = "Date and time"

# Column name of WT Ids in df tables
WT_ID = "WT_ID" 

EVAL_FILE_NAME = "eval.csv"

# used in impute_log from preprocessing step 4
ANY = "__ANY__"
SIGNAL_COL = "Signal"


# used for part 1 of the thesis
START_ANOM = "2019-12-15 00:00:00"
END_ANOM = "2020-04-01 00:00:00"
PATH_PART1_VAL_SET_INJ_DIR = PATH_IMPUTED / "part1" /"part1_val_set_injected_dir"
PATH_PART1_VAL_LOSS_DIR = PATH_IMPUTED / "part1"/ "part1_val_loss_dir"
PATH_PART1_TEST_LOSS_DIR = PATH_IMPUTED / "part1" / "part1_test_loss_dir"
PATH_PART1_K_AGG_METRICS_DIR = PATH_IMPUTED / "part1" / "part1_k_agg_metrics"

# Directory for Hyperparameter tuning
PATH_HPT = PATH_PROJECT_ROOT / "src" / "hyper_parameter_tuning"

#Seeds for Hyperparameter tuning
HP_TUNING_DAY_SEEDS = [1542372833,67927542,2092932925,1283632383,1383562574,230829432,2023702850]

#
PATH_PART2_DETECTIONS = PATH_IMPUTED / "part2" / "detections"
PATH_PART2_EVAL_TEST = PATH_IMPUTED / "part2" / "eval_test_set"
PATH_PART2_RAW_TEST = PATH_IMPUTED / "part2" / "raw_test"
PATH_PART2_WT_FARM = PATH_IMPUTED / "part2" / "wt_farm_pc"
PATH_PART2_EVAL = PATH_IMPUTED / "part2" / "eval"
PATH_PART2_KS_TEST = PATH_IMPUTED / "part2" / "ks_test"
PATH_PART2_KS_TEST_TOTAL = PATH_IMPUTED / "part2" / "ks_test" / "total"