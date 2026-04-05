import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from us_visa.exception import USvisaException
from us_visa.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 1
        self.Denied: int = 0

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocessor
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """ 
        Function accepts raw inputs and then transforms raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data.
        At last it performs prediction using threshold = 0.6
        """
        logging.info("Entered predict method of USvisaModel class")

        try:
            logging.info("Transforming input data using preprocessing object")
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Generating prediction probabilities using trained model")
            y_prob = self.trained_model_object.predict_proba(transformed_feature)[:, 1]

            logging.info("Applying threshold = 0.6 to get final predictions")
            y_pred = (y_prob >= 0.6).astype(int)

            logging.info("Prediction completed successfully")
            return y_pred

        except Exception as e:
            logging.error("Error occurred in predict method")
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"