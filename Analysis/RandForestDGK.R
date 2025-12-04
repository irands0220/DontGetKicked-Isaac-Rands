library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(ranger)
library(themis)

# ============================================================
# 1. READ DATA
# ============================================================
trainData <- vroom("DontGetKicked/OG_Download/training.csv", show_col_types = FALSE) %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

testData <- vroom("DontGetKicked/OG_Download/test.csv", show_col_types = FALSE)

# ============================================================
# 2. CLEAN MMR COLUMNS
# ============================================================
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
# 3. RECIPE
# ============================================================
my_recipe <- recipe(IsBadBuy ~ ., data = trainData) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005) %>%
  step_normalize(all_numeric_predictors())

# ============================================================
# 4. RANDOM FOREST MODEL (TUNEABLE)
# ============================================================
rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 200
) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")

# ============================================================
# 5. WORKFLOW
# ============================================================
rf_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_model)

# ============================================================
# 6. CROSS-VALIDATION FOLDS
# ============================================================
folds <- vfold_cv(trainData, v = 5, repeats = 1)  # smaller v for speed; can increase to 10

# ============================================================
# 7. GRID FOR TUNING
# ============================================================
tuning_grid <- grid_regular(
  mtry(range = c(5, 20)),
  min_n(range = c(5, 30)),
  levels = 2
)

# ============================================================
# 8. TUNE GRID
# ============================================================
set.seed(123)
cv_results <- tune_grid(
  rf_workflow,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc)
)

# ============================================================
# 9. SELECT BEST PARAMETERS
# ============================================================
best_tune <- cv_results %>%
  select_best(metric = "roc_auc")

# ============================================================
# 10. FINAL FIT
# ============================================================
rf_final <- rf_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = trainData)

# ============================================================
# 11. PREDICT ON TEST SET
# ============================================================
preds <- predict(rf_final, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(RefId)) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy = .pred_1)

# ============================================================
# 8. WRITE SUBMISSION FILE
# ============================================================
vroom_write(preds, "/Users/isaacrands/Documents/Stats/Stat_348/DontGetKicked/Submission files/RF_predictions2.csv", delim = ",")
