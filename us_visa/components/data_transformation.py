import sys
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder,
    PowerTransformer, FunctionTransformer
)
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH

# Using a fixed reference year so company_age is consistent across all runs
REFERENCE_YEAR = 2024
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
)
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.estimator import TargetValueMapping


# Education has a natural order — defined once here so it is easy to update
EDUCATION_ORDER = [["High School", "Bachelor's", "Master's", "Doctorate"]]

# Wage-unit to annual multiplier (matches Feature Engineering notebook)
WAGE_MULTIPLIER = {
    'Hour':  2080,   # 52 weeks x 40 hours
    'Week':  52,
    'Month': 12,
    'Year':  1,
}


def _binary_encode(X):
    """
    Convert Y/N columns to 1/0.
    Defined as a module-level function (not lambda) so sklearn can pickle it.
    """
    X = pd.DataFrame(X)
    return X.apply(lambda col: (col == 'Y').astype(int)).values


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        """
        :param data_ingestion_artifact:    Output of data ingestion stage
        :param data_transformation_config: Configuration for this stage
        :param data_validation_artifact:   Output of data validation stage
        """
        try:
            self.data_ingestion_artifact    = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact   = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
        except Exception as e:
            raise USvisaException(e, sys)


    # Static helpers — each mirrors one step from the notebook
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def _fix_negatives(df: pd.DataFrame) -> pd.DataFrame:
        """
        EDA found negative values in no_of_employees — data-entry errors.
        Fix: take absolute value.
        Notebook cell: df["no_of_employees"] = df["no_of_employees"].abs()
        """
        df = df.copy()
        df['no_of_employees'] = df['no_of_employees'].abs()
        logging.info("Fixed negative values in no_of_employees using abs()")
        return df

    @staticmethod
    def _normalize_wage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize prevailing_wage to an annual figure using unit_of_wage,
        store it as prevailing_wage_annual, then drop the original prevailing_wage.

        Notebook cells:
            df['prevailing_wage_annual'] = df['prevailing_wage'] * df['unit_of_wage'].map(multiplier)
            df.drop(columns=['prevailing_wage'], inplace=True)
        """
        df = df.copy()
        df['prevailing_wage_annual'] = (
            df['prevailing_wage'] * df['unit_of_wage'].map(WAGE_MULTIPLIER)
        )
        df.drop(columns=['prevailing_wage'], inplace=True)
        logging.info("Normalized prevailing_wage -> prevailing_wage_annual and dropped original")
        return df

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create company_age = REFERENCE_YEAR - yr_of_estab.
        yr_of_estab is removed later by drop_columns (it is in schema drop_columns).

        Notebook cell:
            df['company_age'] = REFERENCE_YEAR - df['yr_of_estab']
        """
        df = df.copy()
        df['company_age'] = REFERENCE_YEAR - df['yr_of_estab']
        logging.info(f"Created company_age using REFERENCE_YEAR = {REFERENCE_YEAR}")
        return df

    # Preprocessor builder
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Build the ColumnTransformer that exactly matches the Feature Engineering notebook:

            binary  -> FunctionTransformer  (Y/N -> 1/0)
            ordinal -> OrdinalEncoder       (High School < Bachelor's < Master's < Doctorate)
            onehot  -> OneHotEncoder        (drop='first' to avoid multicollinearity)
            num     -> PowerTransformer (yeo-johnson) -> StandardScaler
        """
        try:
            # Read column groups from schema.yaml
            binary_cols  = self._schema_config['binary_columns']   # has_job_experience, full_time_position
            ordinal_cols = self._schema_config['ordinal_columns']  # education_of_employee
            onehot_cols  = self._schema_config['onehot_columns']   # continent, region_of_employment, unit_of_wage
            num_cols     = self._schema_config['num_features']     # no_of_employees, prevailing_wage_annual, company_age

            logging.info(f"Binary cols  : {binary_cols}")
            logging.info(f"Ordinal cols : {ordinal_cols}")
            logging.info(f"OneHot cols  : {onehot_cols}")
            logging.info(f"Numeric cols : {num_cols}")

            # 1. Binary encoder — Y -> 1, N -> 0
            binary_encoder = FunctionTransformer(
                _binary_encode,
                validate=False,
                feature_names_out='one-to-one'
            )

            # 2. Ordinal encoder with meaningful education order
            ordinal_encoder = OrdinalEncoder(
                categories=EDUCATION_ORDER,
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )

            # 3. One-hot encoder — drop='first' to avoid multicollinearity
            onehot_encoder = OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False
            )

            # 4. Numeric pipeline — reduce skew first, then scale
            numeric_pipeline = Pipeline(steps=[
                ('power',  PowerTransformer(method='yeo-johnson')),
                ('scaler', StandardScaler()),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('binary',  binary_encoder,  binary_cols),
                    ('ordinal', ordinal_encoder,  ordinal_cols),
                    ('onehot',  onehot_encoder,   onehot_cols),
                    ('num',     numeric_pipeline, num_cols),
                ],
                remainder='drop'  # safely drops any column not listed above
            )

            logging.info("Preprocessor ColumnTransformer created successfully")
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e
        
    # Main pipeline step

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Full transformation pipeline matching the Feature Engineering notebook:

            1.  Load raw train / test CSVs from ingestion artifact
            2.  Fix negative no_of_employees values
            3.  Normalize prevailing_wage to annual -> prevailing_wage_annual
            4.  Engineer company_age from yr_of_estab
            5.  Separate X / y
            6.  Drop unwanted columns (case_id, yr_of_estab, requires_job_training)
            7.  Encode target: Certified=1, Denied=0
            8.  Fit preprocessor on train, transform both splits
            9.  Apply SMOTE to TRAINING SET ONLY  (test is never resampled)
            10. Stack features + target into numpy arrays
            11. Save preprocessor object and arrays
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")

            # 1. Load raw data
            train_df = DataTransformation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df  = DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Loaded train {train_df.shape} | test {test_df.shape}")

            # 2. Fix negatives in no_of_employees 
            train_df = DataTransformation._fix_negatives(train_df)
            test_df  = DataTransformation._fix_negatives(test_df)

            # 3. Normalize prevailing_wage -> prevailing_wage_annual 
            train_df = DataTransformation._normalize_wage(train_df)
            test_df  = DataTransformation._normalize_wage(test_df)

            # 4. Engineer company_age
            train_df = DataTransformation._engineer_features(train_df)
            test_df  = DataTransformation._engineer_features(test_df)

            # 5. Separate features and target
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test  = test_df.drop(columns=[TARGET_COLUMN])
            y_test  = test_df[TARGET_COLUMN]

            # 6. Drop unwanted columns
            # schema drop_columns covers: case_id, yr_of_estab
            drop_cols = self._schema_config['drop_columns']
            X_train = drop_columns(df=X_train, cols=drop_cols)
            X_test  = drop_columns(df=X_test,  cols=drop_cols)

            # requires_job_training: EDA showed no predictive value → drop
            for split in [X_train, X_test]:
                if 'requires_job_training' in split.columns:
                    split.drop(columns=['requires_job_training'], inplace=True)

            logging.info(f"Features after all drops: {list(X_train.columns)}")

            # 7. Encode target
            # Certified -> 1  |  Denied -> 0   (from TargetValueMapping)
            y_train = y_train.replace(TargetValueMapping()._asdict())
            y_test  = y_test.replace(TargetValueMapping()._asdict())

            # 8. Preprocess 
            preprocessor = self.get_data_transformer_object()

            X_train_arr = preprocessor.fit_transform(X_train)
            logging.info("Preprocessor fitted on train and transformed")

            X_test_arr  = preprocessor.transform(X_test)
            logging.info("Preprocessor transform applied to test set")

            #  9. SMOTE — training set ONLY 
            # Applying SMOTE to test data would leak synthetic distribution
            # into evaluation — the test set must reflect real-world data.
            logging.info("Applying SMOTE to training set only")
            smote = SMOTE(random_state=42)
            X_train_final, y_train_final = smote.fit_resample(X_train_arr, y_train)
            logging.info(
                f"After SMOTE: train shape {X_train_final.shape} | "
                f"class counts {np.bincount(y_train_final.astype(int))}"
            )

            # Test set stays as-is
            X_test_final  = X_test_arr
            y_test_final  = y_test

            # 10. Stack features + target into final arrays 
            train_arr = np.c_[X_train_final, np.array(y_train_final)]
            test_arr  = np.c_[X_test_final,  np.array(y_test_final)]

            logging.info(f"Final train array shape : {train_arr.shape}")
            logging.info(f"Final test  array shape : {test_arr.shape}")

            # 11. Save artifacts
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            logging.info("Saved preprocessor object and transformed numpy arrays")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e