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

    def get_data_from_input(self, data):
        column_names = ['LoanNr_ChkDgt', 'Name', 'City', 'State', 'Zip', 'Bank', 
                        'BankState', 'NAICS', 'ApprovalDate', 'ApprovalFY', 
                        'Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob',
                        'FranchiseCode', 'UrbanRural', 'RevLineCr', 'LowDoc', 
                        'ChgOffDate', 'DisbursementDate', 'DisbursementGross', 
                        'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']

        loans_predict = pd.DataFrame(data, columns=column_names)

        return loans_predict

    def get_data_from_csv(self):
        loans = pd.read_csv(self.config.data_path, low_memory=False)

        return loans

    def clean_and_preprocess_data(self, data=None):
        # Check if the data is provided as input or from CSV
        if isinstance(data, pd.DataFrame):
            loans = data  # Use the DataFrame directly if provided
        elif data is None:
            loans = self.get_data_from_csv()  # Load from CSV if no data provided
        else:
            # If data is not a DataFrame or None, try to create a DataFrame from it
            loans = self.get_data_from_input(data)

        # Drop null values from specified columns
        loans.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist', 'RevLineCr', 'LowDoc', 'DisbursementDate'], inplace=True)
        
        if isinstance(data, pd.DataFrame) or data is None:
            loans.dropna(subset=['MIS_Status'], inplace=True)
        else:
            print()

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

        loans['FranchiseCode'] = pd.to_numeric(loans['FranchiseCode'], errors='coerce')

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

        loans['Term'] = pd.to_numeric(loans['Term'], errors='coerce')
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
            'LowDoc': 'int64',
            'RetainedJob': 'int64',
            'CreateJob': 'int64',
            'NoEmp': 'int64',
        })

        # Create Default column based on MIS_Status
        if isinstance(data, pd.DataFrame) or data is None:
            loans['Default'] = np.where(loans['MIS_Status'] == 'P I F', 0, 1)
        else:
            print()

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
        loans.drop(columns=['LoanNr_ChkDgt', 'ChgOffDate', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'NewExist', 'FranchiseCode',
                      'ApprovalDate', 'DisbursementDate', 'Industry'], inplace=True)
        
        if isinstance(data, pd.DataFrame) or data is None:
            loans.drop(columns=['MIS_Status'], inplace=True)
        else:
            print()

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

        logger.info("clean_and_preprocess_data SUCCESS!!")
        return loans
    

    
    def transform_data(self, loans, isPredict=False):
        loans_tf = loans

        # Apply log1p transformation to the selected features
        features_transform = ['DaysToDisbursement', 'DisbursementGross',
                            'RetainedJob', 'CreateJob', 'NoEmp', 'Term']
        loans_tf[features_transform] = np.log1p(loans_tf[features_transform])

        # Create dummies for the categorical fields in the prediction data
        if isPredict:
            # Generate one-hot encoded dummies for the test data
            loans_tf = pd.get_dummies(loans_tf, columns=['State', 'BankState', 'RevLineCr', 'LowDoc'])

            # Load training columns to ensure consistent structure
            train_columns = pd.read_csv(self.config.train_columns, header=None)[0].tolist()

            # Reindex to align with training columns, filling missing columns with 0
            loans_tf = loans_tf.reindex(columns=train_columns, fill_value=0)
            loans_tf.drop(loans_tf.columns[0], axis=1, inplace=True)
            loans_tf.drop(columns='Default', inplace=True)
            loans_tf.replace(True, 1, inplace=True)
            loans_tf = loans_tf.astype(float)

            # Renaming columns to just numbers
            loans_tf.columns = [i for i, _ in enumerate(loans_tf.columns)]

            pd.DataFrame(loans_tf).to_csv(os.path.join(self.config.root_dir, "test_columns.csv"), index=False)
        else:
            # Apply dummies for the training data
            loans_tf = pd.get_dummies(loans_tf, columns=['State', 'BankState', 'RevLineCr', 'LowDoc'])

            # Drop rows with NaN in 'DaysToDisbursement'
            loans_tf.dropna(subset=['DaysToDisbursement'], inplace=True)

            # Replace infinite values with NaN
            loans_tf.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Optionally drop rows with NaN values
            loans_tf.dropna(inplace=True)

            # Save the training columns to CSV for future use
            pd.DataFrame(loans_tf.columns).to_csv(os.path.join(self.config.root_dir, "train_columns.csv"), index=False)

        logger.info(loans_tf.shape)
        logger.info("transform_data SUCCESS!!")


        return loans_tf




    def train_test_spliting(self, loans_cleaned):
        X = loans_cleaned.drop(columns='Default')
        y = loans_cleaned[['Default']]

        # Scale predictors for easier training
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pd.DataFrame(X_scaled).to_csv(os.path.join(self.config.root_dir, "X_scaled.csv"), index=False)

        # first split: train & pretest (split for validation & test)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Save the train and test sets to CSV files
        pd.DataFrame(X_train).to_csv(os.path.join(self.config.root_dir, "X_train.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(self.config.root_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)


       # Log detailed information about splits
        logger.info("Data successfully split into training, validation, and test sets.")
        logger.info(f"Training set X shape: {X_train.shape}, y shape: {y_train.shape}")
        logger.info(f"Test set X shape: {X_test.shape}, y shape: {y_test.shape}")

        # Print summary of the splits
        print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
