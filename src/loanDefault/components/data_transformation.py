import os
from src.loanDefault import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing
from src.loanDefault.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def clean_and_preprocess_data(self):
        # Read the CSV file
        loans = pd.read_csv(self.config.data_path, low_memory=False)

        # Drop null values from specified columns
        loans.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist', 'RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)

        # Define a function to clean the string from integer data
        def clean_data_str(value):
            if isinstance(value, str):
                return value.replace('A', '')
            return value

        loans['ApprovalFY'] = loans['ApprovalFY'].apply(clean_data_str).astype('int64')

        # Remove '$', commas, and extra spaces from records in columns with dollar values that should be floats
        loans[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = \
        loans[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].apply(lambda x: x.str.replace('$', '').str.replace(',', '').str.strip())
        
        # Reconfigure remaining data into appropriate data types
        loans = loans.astype({
            'Zip': 'str',
            'NewExist': 'int64',
            'UrbanRural': 'str',
            'DisbursementGross': 'float64',
            'BalanceGross': 'float64',
            'ChgOffPrinGr': 'float64',
            'GrAppv': 'float64',
            'SBA_Appv': 'float64'
        })

        # Mapping dictionary for NAICS codes
        naics_mapping = {
            '11': 'Agri/For/Fish/Hunt', # 'Agriculture, forestry, fishing and hunting',
            '21': 'Min/Quar/Oil/Gas_ext', # 'Mining, quarrying, and oil and gas extraction',
            '22': 'Utilities', # 'Utilities',
            '23': 'Construction', # 'Construction',
            '31': 'Manufacturing', # 'Manufacturing',
            '32': 'Manufacturing', # 'Manufacturing',
            '33': 'Manufacturing', # 'Manufacturing',
            '42': 'Wholesale_trade', # 'Wholesale trade',
            '44': 'Retail_trade', # 'Retail trade',
            '45': 'Retail_trade', # 'Retail trade',
            '48': 'Transp/WareH', # 'Transportation and warehousing',
            '49': 'Transp/WareH', # 'Transportation and warehousing',
            '51': 'Info', # 'Information',
            '52': 'Finance/Insurance', # 'Finance and insurance',
            '53': 'Real_est/Rental/Lease', # 'Real estate and rental and leasing',
            '54': 'Prof/Science/Tech', # 'Professional, scientific, and technical services',
            '55': 'Mgmt_comp_entp', # 'Management of companies and enterprises',
            '56': 'Admin/Supp/WasteM/Remed', # 'Administrative and support and waste management and remediation services',
            '61': 'Edu', # 'Educational services',
            '62': 'Health/Social', # 'Health care and social assistance',
            '71': 'Art/Entr/Rec', # 'Arts, entertainment, and recreation',
            '72': 'Accom/Food', # 'Accommodation and food services',
            '81': 'Other', # 'Other services (except public administration)',
            '92': 'Pub_Admin', # 'Public administration'
        }

        # Apply NAICS mapping
        loans['Industry'] = loans['NAICS'].astype('str').apply(lambda x: x[:2])
        loans['IndustryCode'] = loans['NAICS'].astype('str').apply(lambda x: x[:2])
        loans['Industry'] = loans['Industry'].map(naics_mapping)
        loans.dropna(subset=['Industry'], inplace=True)

        loans = loans[(loans['NewExist'] == 1) | (loans['NewExist'] == 2)]

        # Create NewBusiness column
        loans.loc[loans['NewExist'] == 1, 'NewBusiness'] = 0
        loans.loc[loans['NewExist'] == 2, 'NewBusiness'] = 1

        # Filter and clean RevLineCr and LowDoc columns
        loans = loans[(loans['RevLineCr'] == 'Y') | (loans['RevLineCr'] == 'N')]
        loans = loans[(loans['LowDoc'] == 'Y') | (loans['LowDoc'] == 'N')]

        pd.set_option('future.no_silent_downcasting', True)
        loans['RevLineCr'] = loans['RevLineCr'].replace(['N', 'Y'], [0, 1])
        loans['LowDoc'] = loans['LowDoc'].replace(['N', 'Y'], [0, 1])

        # Create IsFranchise flag
        loans.loc[loans['FranchiseCode'] <= 1, 'IsFranchise'] = 0
        loans.loc[loans['FranchiseCode'] > 1, 'IsFranchise'] = 1

        # Convert date columns to datetime
        loans[['ApprovalDate', 'DisbursementDate']] = loans[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)
        loans['DaysToDisbursement'] = (loans['DisbursementDate'] - loans['ApprovalDate']).dt.days.astype('int64')

        # Create DaysToDisbursement column calculating days between DisbursementDate and ApprovalDate
        loans['DaysToDisbursement'] = (loans['DisbursementDate'] - loans['ApprovalDate']).dt.days

        # Convert DaysToDisbursement to int64 dtype
        loans['DaysToDisbursement'] = loans['DaysToDisbursement'].astype('int64')

        # Create DisbursementMonth & DisbursementYear field for time analysis later (Great Recession categorizing)
        loans['DisbursementMonth'] = loans['DisbursementDate'].map(lambda x: x.month)
        loans['DisbursementYear'] = loans['DisbursementDate'].map(lambda x: x.year)

        # Additional preprocessing steps (StateSame, SBA_AppvPercent, etc.)
        loans['StateSame'] = np.where(loans['State'] == loans['BankState'], 1, 0)
        loans['SBA_AppvPercent'] = loans['SBA_Appv'] / loans['GrAppv']
        loans['AppvDisbursed'] = np.where(loans['DisbursementGross'] == loans['GrAppv'], 1, 0)
        loans['RealEstate'] = np.where(loans['Term'] >= 240, 1, 0)
        loans['GreatRecession'] = np.where(
            (loans['DisbursementYear'] == 2007) & (loans['DisbursementMonth'] >= 12) |
            (loans['DisbursementYear'] == 2008) |
            (loans['DisbursementYear'] == 2009) & (loans['DisbursementMonth'] <= 6),
            1, 0)

        loans = loans.astype({
            'IsFranchise': 'int64',
            'NewBusiness': 'int64',
            'RevLineCr': 'int64',
            'LowDoc': 'int64'
        })

        # Create Default column based on MIS_Status
        loans['Default'] = np.where(loans['MIS_Status'] == 'P I F', 0, 1)

        state_mapping = {state: idx for idx, state in enumerate([
            'IN', 'OK', 'FL', 'CT', 'NJ', 'NC', 'IL', 'RI', 'TX', 'VA',
            'TN', 'AR', 'MN', 'MO', 'MA', 'CA', 'SC', 'LA', 'IA', 'OH',
            'KY', 'MS', 'NY', 'MD', 'PA', 'OR', 'ME', 'KS', 'MI', 'AK',
            'WA', 'CO', 'MT', 'WY', 'UT', 'NH', 'WV', 'ID', 'AZ', 'NV',
            'WI', 'NM', 'GA', 'ND', 'VT', 'AL', 'NE', 'SD', 'HI', 'DE', 'DC'
        ])}

        bank_state_mapping = {state: idx for idx, state in enumerate([
            'OH', 'IN', 'OK', 'FL', 'DE', 'SD', 'AL', 'CT', 'GA', 'OR', 'MN', 'RI', 'NC', 'TX',
            'MD', 'NY', 'TN', 'SC', 'MS', 'MA', 'LA', 'IA', 'VA', 'CA', 'IL', 'KY', 'PA', 'MO',
            'WA', 'MI', 'UT', 'KS', 'WV', 'WI', 'AZ', 'NJ', 'CO', 'ME', 'NH', 'AR', 'ND', 'MT',
            'ID', 'WY', 'NM', 'DC', 'NV', 'NE', 'PR', 'HI', 'VT', 'AK', 'GU', 'AN', 'EN', 'VI'
        ])}

        # Drop unwanted columns
        loans.drop(columns=['LoanNr_ChkDgt', 'ChgOffDate', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'MIS_Status', 'NewExist', 'FranchiseCode',
                      'ApprovalDate', 'DisbursementDate', 'Industry'], inplace=True)
        
        # Encoding like base model encoding
        loans['State'] = loans['State'].map(state_mapping)
        loans['BankState'] = loans['BankState'].map(bank_state_mapping)
        loans = loans.astype({
            'IndustryCode' : 'int64',
            'UrbanRural': 'int64'
            })
        
        # Drop columns based on VIF Score calculation
        loans.drop(columns=['UrbanRural', 'RealEstate', 'DisbursementYear', 'GrAppv', 'SBA_Appv', 'ChgOffPrinGr', 'IndustryCode'], inplace=True)

        # drop null values after cleaning & processing
        loans.dropna(subset=['DaysToDisbursement'], inplace=True)

        # Return the cleaned and preprocessed loans dataframe
        return loans
    

    
    def transform_data(self, loans):
        loans_tf = loans

        features_transform = ['DaysToDisbursement', 'DisbursementGross',
                            'RetainedJob', 'CreateJob', 'NoEmp', 'Term']
        
        loans_tf[features_transform] = np.log1p(loans_tf[features_transform])

        loans_tf = pd.get_dummies(loans_tf)

        loans_tf.dropna(subset=['DaysToDisbursement'], inplace=True)

        # Replace infinite values with NaN
        loans_tf.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Optionally drop rows with NaN values
        loans_tf.dropna(inplace=True)

        # One-Hot Encoding for non ranking feature fields
        loans_tf = pd.get_dummies(loans_tf, columns=['State']).astype(int)
        loans_tf = pd.get_dummies(loans_tf, columns=['BankState']).astype(int)
        loans_tf = pd.get_dummies(loans_tf, columns=['RevLineCr']).astype(int)
        loans_tf = pd.get_dummies(loans_tf, columns=['LowDoc']).astype(int)

        return loans_tf




    def train_test_spliting(self, loans_cleaned):
        X = loans_cleaned.drop(columns='Default')
        y = loans_cleaned[['Default']]

        # Scale predictors for easier training
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # first split: train & pretest (split for validation & test)
        X_train, X_pretest, y_train, y_pretest = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # second split: validation & test (validation for hyperparameter tuning)
        X_val, X_test, y_val, y_test = train_test_split(X_pretest, y_pretest, test_size=0.50, random_state=42)

        # Save the train and test sets to CSV files
        pd.DataFrame(X_train).to_csv(os.path.join(self.config.root_dir, "X_train.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
        pd.DataFrame(X_val).to_csv(os.path.join(self.config.root_dir, "X_val.csv"), index=False)
        pd.DataFrame(y_val).to_csv(os.path.join(self.config.root_dir, "y_val.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(self.config.root_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)


       # Log detailed information about splits
        logger.info("Data successfully split into training, validation, and test sets.")
        logger.info(f"Training set X shape: {X_train.shape}, y shape: {y_train.shape}")
        logger.info(f"Validation set X shape: {X_val.shape}, y shape: {y_val.shape}")
        logger.info(f"Test set X shape: {X_test.shape}, y shape: {y_test.shape}")

        # Print summary of the splits
        print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Validation data shape: X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
