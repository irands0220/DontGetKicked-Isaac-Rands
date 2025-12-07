library(tidyverse)
library(tidymodels)
library(vroom)
library(discrim)   # For naive_Bayes()
library(themis)    # For step_upsample
library(embed)     # For step_other

# ------------------------------------------------------------
# 0. SETUP
# ------------------------------------------------------------
set.seed(348)

# ============================================================
# 1. READ AND CLEAN DATA
# ============================================================
# --- READ DATA ---
trainData <- vroom("DontGetKicked/OG_Download/training.csv", show_col_types = FALSE) %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

testData <- vroom("DontGetKicked/OG_Download/test.csv", show_col_types = FALSE)

# --- CLEAN MMR COLUMNS ---
clean_columns <- c(
  "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice",
  "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice",
  "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice",
  "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice"
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
nb_recipe <- recipe(IsBadBuy ~ ., data = trainData) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_normalize(all_predictors(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%           # Remove zero-variance predictors
  step_upsample(IsBadBuy, over_ratio = 1)

# ============================================================
# 3. MODEL
# ============================================================
nb_model <- naive_Bayes(
  Laplace = tune(),       # no range here
  smoothness = tune()     # no range here
) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# ============================================================
# 4. WORKFLOW
# ============================================================
nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

# ============================================================
# 5. RESAMPLING
# ============================================================
folds <- vfold_cv(trainData, v = 10, repeats = 2)

# ============================================================
# 6. TUNING GRID
# ============================================================
nb_grid <- grid_regular(
  Laplace(range = c(0.1, 1)),       # define range in grid
  smoothness(range = c(0.1, 1)),    # define range in grid
  levels = 5
)

# ============================================================
# 7. TUNING
# ============================================================
nb_cv_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = nb_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(save_pred = TRUE)
  )

# ============================================================
# 8. FINAL FIT
# ============================================================
nb_best <- nb_cv_results %>%
  select_best(metric = "roc_auc")
nb_fit <- nb_wf %>%
  finalize_workflow(nb_best) %>%
  fit(data = trainData)

# ============================================================
# 9. PREDICTIONS
# ============================================================
preds <- nb_fit %>%
  predict(new_data = testData, type = "prob")

preds_submission <- preds %>%
  select(.pred_1) %>%
  bind_cols(testData %>% select(RefId), .) %>%
  rename(IsBadBuy = .pred_1) %>%
  select(RefId, IsBadBuy)

# ============================================================
# 10. SAVE SUBMISSION
# ============================================================
vroom_write(
  x = preds_submission,
  file = "/Users/isaacrands/Documents/Stats/Stat_348/DontGetKicked/Submission files/NB_predictionsFinal.csv",
  delim = ","
)
