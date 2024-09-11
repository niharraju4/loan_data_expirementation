import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MODEL_NAME = 'classification.pkl'

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TARGET = 'Loan_Status'

FEATURES =['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'Property_Area']

NUMERICAL_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CATEGORICAL_FEATURES = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']

# IN MY CASE IT IS SAME AS CATEGRROICAL FEATURES
FEATUREES_TO_ENCODE = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURES_TO_ADD = 'CoApplicantIncome'

DROP_FEATURES = ['CoApplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']