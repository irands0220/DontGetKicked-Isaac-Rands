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
my_recipe <- recipe(type ~ ., data = trainData) %>%
  update_role(id, new_role = "ID") %>%  # exclude ID from predictors
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%  # target encoding for categorical vars
  step_normalize(all_numeric_predictors())  # standardize numeric predictors

# --- MODEL ---
nb_model <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("klaR", usekernel = TRUE)

# --- WORKFLOW ---
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# --- FIT MODEL ---
nb_fit <- nb_wf %>%
  fit(data = trainData)

# --- PREDICT ON TEST DATA ---
preds <- nb_fit %>%
  predict(new_data = testData, type = "class") %>%
  rename(type = .pred_class)

# --- BIND WITH ID safely ---
preds <- bind_cols(
  id = testData$id,
  preds
)

# --- WRITE SUBMISSION FILE ---
vroom_write(
  x = preds,
  file = "/Users/isaacrands/Documents/Stats/Stat_348/GGG_Comp/Submissions/NB_GGG_submission.csv",
  delim = ","
)
