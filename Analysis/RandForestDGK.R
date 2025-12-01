library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(glmnet)
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


# --- MODEL SPEC ---
rf_bal <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# --- WORKFLOW ---
balanced_workflow <- workflow() %>%
  add_model(rf_bal) %>%
  add_recipe(my_recipe)

# --- GRID ---
tuning_grid <- grid_regular(mtry(range=c(1, 9)), min_n(), levels = 3)

# --- FOLDS ---
folds <- vfold_cv(trainData, v = 10, repeats = 1)

# --- TUNE ---
cv_results <- tune_grid(
  balanced_workflow,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc)
)

best_tune <- cv_results %>% select_best(metric = "roc_auc")

# --- FINAL FIT ---
rand_for_fit <- balanced_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = trainData)

# --- PREDICT ---
preds <- predict(rand_for_fit, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(RefId)) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy = .pred_1)


vroom_write(x=preds, file="./NaiveBayes_predictions.csv", delim=",")
