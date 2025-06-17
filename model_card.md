# Model Card: Census Income Classification

## Model Summary

This model predicts whether a person's income is above or below $50K per year, using U.S. Census data. 

## Dataset

- **Source:** U.S. Census Bureau (Adult Census Income dataset)
- **Features:** age, education, marital status, occupation, relationship, race, sex, native country, and others.
- **Target:** income (`<=50K` or `>50K`)

## Model Details

- **Type:** RandomForestClassifier (Scikit-learn)
- **Language/Framework:** Python 3.8, Scikit-learn, FastAPI
- **Training Date:** June 2025

## Model Performance

### Overall test set:

- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

## Slice Performance (Selected Examples)

### workclass feature:
- **Private**: Precision 0.7376 | Recall 0.6404 | F1 0.6856
- **Federal-gov**: Precision 0.7971 | Recall 0.7857 | F1 0.7914
- **Self-emp-not-inc**: Precision 0.7064 | Recall 0.4904 | F1 0.5789

### education feature:
- **Bachelors**: Precision 0.7523 | Recall 0.7289 | F1 0.7404
- **Masters**: Precision 0.8271 | Recall 0.8551 | F1 0.8409
- **HS-grad**: Precision 0.6594 | Recall 0.4377 | F1 0.5261

### marital-status feature:
- **Never-married**: Precision 0.8302 | Recall 0.4272 | F1 0.5641
- **Married-civ-spouse**: Precision 0.7346 | Recall 0.6900 | F1 0.7116
- **Divorced**: Precision 0.7600 | Recall 0.3689 | F1 0.4967

(Full slice outputs available in `slice_output.txt`.)

## How the model was trained

Data was cleaned and encoded, then split 80/20 into train and test sets. The model uses default hyperparameters for RandomForestClassifier. Preprocessing includes one-hot encoding for categorical features.

## Intended Use

This model is for demonstration only. Any commercial or research use would require extensive additional testing for bias, fairness, and accuracy.

## Limitations

- This model was trained on U.S. Census data from one point in time.
- It may not work as well on different populations or new data.
- I did not do any fairness or bias analysis beyond accuracy metrics.

## Maintenance

The model does not include automatic monitoring or retraining. 

