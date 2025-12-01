# --- LIBRARIES ---
library(tidyverse)
library(tidymodels)
library(vroom)
library(discrim)
library(doParallel)
library(themis)
library(klaR)

# --- READ DATA ---
trainData <- vroom("DontGetKicked/OG_Download/training.csv", show_col_types = FALSE) %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

testData <- vroom("DontGetKicked/OG_Download/test.csv", show_col_types = FALSE)

# --- RECIPE ---
my_recipe <- recipe(IsBadBuy ~ ., data = trainData) %>%
  step_unknwon(all_nominal_predictors())
  step_dummy(all_nominal_predictors())

# --- MODEL ---
nb_model <- naive_Bayes(
  Laplace = tune(),
  smoothness = tune()
) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# --- WORKFLOW ---
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# --- RESAMPLING ---
folds <- vfold_cv(trainData, v = 10, repeats = 2)

# --- TUNING GRID ---
nb_grid <- grid_regular(
  Laplace(),
  smoothness(),
  levels = 5
)

# --- TUNING ---
nb_cv_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = nb_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(save_pred = TRUE, parallel_over = "everything")
  )

# --- SELECT BEST PARAMETERS ---
nb_best <- nb_cv_results %>%
  select_best(metric = "roc_auc")

# --- FINALIZE AND FIT WORKFLOW ---
nb_fit <- nb_wf %>%
  finalize_workflow(nb_best) %>%
  fit(data = trainData)

preds <- nb_fit %>% 
  predict(new_data=testData, type="prob")

preds <- preds %>% 
  select(.pred_1) %>% 
  bind_cols(., testData) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep resource and predictions
  rename(Action=.pred_1)

#  
vroom_write(x=preds, file="./NaiveBayes_predictions.csv", delim=",")

# --- STOP PARALLEL CLUSTER ---
stopCluster(cl)
