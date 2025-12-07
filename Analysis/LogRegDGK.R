library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(ranger)
library(themis)

# ============================================================
# 1. READ DATA (Same as before)
# ============================================================
trainData <- vroom("DontGetKicked/OG_Download/training.csv", show_col_types = FALSE) %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

testData <- vroom("DontGetKicked/OG_Download/test.csv", show_col_types = FALSE)

# ============================================================
# 2. CLEAN MMR COLUMNS (Same as before)
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
# 3. RECIPE (Same as before)
# ============================================================
my_recipe <- recipe(IsBadBuy ~ ., data = trainData) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_upsample(IsBadBuy, over_ratio = 1) # Retaining upsampling for class balance

# ============================================================
# 4. LOGISTIC REGRESSION MODEL
# ============================================================
log_reg_model <- logistic_reg() %>%
  set_mode("classification") %>%
  set_engine("glm") # glm is the standard engine for logistic regression

# ============================================================
# 5. WORKFLOW
# ============================================================
log_reg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_reg_model)

# ============================================================
# 6. FINAL FIT (No tuning required for standard logistic regression)
# ============================================================
log_reg_final <- log_reg_workflow %>%
  fit(data = trainData)

# ============================================================
# 7. PREDICT ON TEST SET
# ============================================================
preds_logreg <- predict(log_reg_final, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(RefId)) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy = .pred_1)

# ============================================================
# 8. WRITE SUBMISSION FILE
# ============================================================
vroom_write(preds_logreg, "/Users/isaacrands/Documents/Stats/Stat_348/DontGetKicked/Submission files/LogReg_predictions.csv", delim = ",")

