library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(themis)
library(dbarts)

# ============================================================
# 1. READ + CLEAN DATA
# ============================================================
trainData <- vroom("DontGetKicked/OG_Download/training.csv", show_col_types = FALSE) %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

testData <- vroom("DontGetKicked/OG_Download/test.csv", show_col_types = FALSE)

clean_columns <- c(
  "MMRAcquisitionAuctionAveragePrice",
  "MMRAcquisitionAuctionCleanPrice",
  "MMRAcquisitionRetailAveragePrice",
  "MMRAcquisitonRetailCleanPrice",
  "MMRCurrentAuctionAveragePrice",
  "MMRCurrentAuctionCleanPrice",
  "MMRCurrentRetailAveragePrice",
  "MMRCurrentRetailCleanPrice"
)

fix_mmr_cols <- function(df) {
  df %>%
    mutate(across(all_of(clean_columns), as.character)) %>%
    mutate(across(all_of(clean_columns), ~ na_if(., "NULL"))) %>%
    mutate(across(all_of(clean_columns), ~ na_if(., ""))) %>%
    mutate(across(all_of(clean_columns), as.numeric))
}

trainData <- fix_mmr_cols(trainData)
testData  <- fix_mmr_cols(testData)

# ============================================================
# 2. RECIPE
# ============================================================
bart_recipe <- recipe(IsBadBuy ~ ., data = trainData) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_upsample(IsBadBuy, over_ratio = 1)

# ============================================================
# 3. BART MODEL
# ============================================================
bart_model <- bart() %>%
  set_engine("dbarts", ntree = 200) %>%  # fixed number of trees
  set_mode("classification")

# ============================================================
# 4. WORKFLOW
# ============================================================
bart_workflow <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_model)

# ============================================================
# 5. PARAMETER EXTRACTION & GRID (here only placeholder)
# ============================================================
# BART doesn't expose tunable parameters in tidymodels for classification,
# so we just create a dummy grid with 1 row to mimic the old workflow
forest_params <- extract_parameter_set_dials(bart_workflow)
forest_params <- finalize(forest_params, trainData)
my_grid <- grid_regular(forest_params, levels = 1)

# ============================================================
# 6. 5-FOLD CROSS-VALIDATION
# ============================================================
folds <- vfold_cv(trainData, v = 5)

# CV results (optional, mainly for consistent workflow)
CV_results <- bart_workflow %>%
  fit_resamples(
    resamples = folds,
    metrics = metric_set(roc_auc)
  )

# ============================================================
# 7. FINAL FIT
# ============================================================
final_wf <- bart_workflow %>%
  fit(data = trainData)

# ============================================================
# 8. PREDICT ON TEST SET
# ============================================================
bart_preds <- predict(final_wf, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(RefId)) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy = .pred_1)

# ============================================================
# 9. WRITE SUBMISSION FILE
# ============================================================
vroom_write(
  bart_preds,
  "/Users/isaacrands/Documents/Stats/Stat_348/DontGetKicked/Submission files/BART_predictions.csv",
  delim = ","
)
