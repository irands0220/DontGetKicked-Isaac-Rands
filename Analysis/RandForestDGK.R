library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(ranger)
library(themis)

# --- READ DATA ---
trainData <- vroom("DontGetKicked/OG_Download/training.csv", show_col_types = FALSE) %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

testData <- vroom("DontGetKicked/OG_Download/test.csv", show_col_types = FALSE)

# --- RECIPE ---
my_recipe <- recipe(IsBadBuy ~ ., data = trainData) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005) %>%
  step_normalize(all_numeric_predictors())


# --- RANDOM FOREST MODEL (NO TUNING) ---
rf_model <- rand_forest(
  mtry = 7,
  min_n = 10,
  trees = 100
) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")


# --- WORKFLOW ---
rf_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_model)


# --- FIT FINAL MODEL ---
rf_fit <- rf_workflow %>%
  fit(data = trainData)


# --- PREDICT ON TEST SET ---
preds <- predict(rf_fit, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(RefId)) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy = .pred_1)


# --- WRITE SUBMISSION FILE ---
vroom_write(preds, "RF_predictions.csv", delim = ",")
