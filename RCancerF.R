library(tidyverse)
library(MASS)

cancer = read_csv("~/GitHub/BreastCancerAI/train_data.csv",
                  col_names = FALSE)
names(cancer) = c('diagnosis', 'radius_mean', 'texture_mean',
      'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
      'concavity_mean', 'concave_points_mean', 'symmetry_mean',
      'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
      'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
      'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
      'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
      'smoothness_worst', 'compactness_worst', 'concavity_worst',
      'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst')

# cancer <- read_csv("HackAI/cancer.csv")

null = glm(diagnosis ~ 1, family = binomial, data = cancer)
full = glm(diagnosis ~ radius_mean + texture_mean + 
      perimeter_mean + area_mean + smoothness_mean + compactness_mean +
      concavity_mean + concave_points_mean + symmetry_mean + 
      fractal_dimension_mean + radius_se + texture_se + perimeter_se + 
      area_se + smoothness_se + compactness_se + concavity_se + 
      concave_points_se + symmetry_se + fractal_dimension_se +
      radius_worst + texture_worst + perimeter_worst + area_worst + 
      smoothness_worst + compactness_worst + concavity_worst + 
      concave_points_worst + symmetry_worst + fractal_dimension_worst, 
      family = binomial, data = cancer)

# stepf = stepAIC(null, scope = list(lower = null, upper = full),
#                 direction = 'forward', trace = F)
stepboth = stepAIC(null, scope = list(lower = null, upper = full),
                   direction = 'both', trace = F)

stepboth

full = glm(diagnosis ~ perimeter_worst * smoothness_worst * texture_mean * 
             area_se * symmetry_worst * compactness_se * concavity_mean * 
             concave_points_worst * compactness_mean, 
           family = binomial, data = cancer)

stepboth = stepAIC(null, scope = list(lower = null, upper = full),
                   direction = 'both', trace = F)
stepboth

test <- read_csv("~/GitHub/BreastCancerAI/test_data.csv",
                 col_names = FALSE)
names(test) = c('diagnosis', 'radius_mean', 'texture_mean',
                  'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                  'concavity_mean', 'concave_points_mean', 'symmetry_mean',
                  'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
                  'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                  'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                  'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                  'smoothness_worst', 'compactness_worst', 'concavity_worst',
                  'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst')

pred = predict(stepboth, newdata = test, type = 'response')
pred = ifelse(pred > 0.5, 1, 0)
true_val = test$diagnosis

tp = sum(ifelse(pred == 1 & true_val == 1, 1, 0))
tn = sum(ifelse(pred == 0 & true_val == 0, 1, 0))
fn = sum(ifelse(pred == 1 & true_val == 0, 1, 0))
fp = sum(ifelse(pred == 0 & true_val == 1, 1, 0))

acc = (tn + tp)/ 114
err = 1 - acc
err * 100
