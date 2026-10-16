# kaggle-entity-resolution

# Learnings
- Simple features are very effective at isolating matches from random negative assignments; initial features used are 
1/ exact matches for addresses and business names 2/ length of business names and addresses 3/ levenshtein distance 
between names and addresses 4/ number of matching names between names and addresses
- With a simple model the F1-score if 99.6% we may have to refine evaluation to harder test cases to drive improvement.
Initial ideas are a subset of "Hard" cases and tracking the count of FP and FN in a true hold out test case
- With 22 million possible match combinations computation efficiency is as important as model accuracy. I am adding 
the run time cost to the model
- Random Forest Predictor is too slow for model iterations, switching to Logistic Regression with minimal loss in accuracy
- Using raw coefficients over Sklearn model improves modeling speed by more than 10x
- Using sorted entity list and match list (by business name) and starting search by using first letter of 
the name improves the mean search distance to ~50K candidates before finding a match. This still means that
we would have to search 500B matches before completion
