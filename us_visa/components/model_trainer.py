import os
import sys
from typing import Tuple

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data, load_object, save_object
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
)
from us_visa.entity.estimator import USvisaModel

THRESHOLD = 0.6


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # ------------------------------------------------------------------
    # Train XGBoost
    # ------------------------------------------------------------------

    def train_xgboost(self, x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """
        Train XGBoost with scale_pos_weight computed from the training labels.

        From the Feature Engineering notebook:
            - XGBoost was selected as the best model based on recall & F1.
            - scale_pos_weight = certified_count / denied_count handles class
              imbalance directly inside XGBoost (complementing SMOTE).
            - eval_metric='logloss', random_state=42.
        """
        try:
            logging.info("Computing scale_pos_weight from training labels")
            certified_count = int((y_train == 1).sum())
            denied_count    = int((y_train == 0).sum())
            scale_pos_weight = certified_count / denied_count

            logging.info(
                f"Certified (1): {certified_count} | "
                f"Denied (0): {denied_count} | "
                f"scale_pos_weight: {round(scale_pos_weight, 4)}"
            )

            model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=42,
                use_label_encoder=False,
            )

            logging.info("Training XGBoost model")
            model.fit(x_train, y_train)
            logging.info("XGBoost training complete")

            return model

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # Evaluate model
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: XGBClassifier,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[ClassificationMetricArtifact, float]:
        """
        Evaluate on the test set and return the metric artifact plus recall score.

        Recall is the PRIMARY metric for this problem:
            - We want to correctly identify as many Certified cases as possible.
            - A missed Certified case (false negative) is more costly than a
              false alarm.
        All four metrics are logged so the full picture is available in the logs.
        """
        try:
            y_prob = model.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= THRESHOLD).astype(int)

            acc       = accuracy_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)

            logging.info("=== Model Evaluation on Test Set ===")
            logging.info(f"  Accuracy  : {round(acc,       4)}")
            logging.info(f"  F1 Score  : {round(f1,        4)}")
            logging.info(f"  Precision : {round(precision, 4)}")
            logging.info(f"  Recall    : {round(recall,    4)}  ← primary metric")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
            )
            return metric_artifact, recall

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Full model training pipeline:

            1. Load transformed train / test numpy arrays from transformation artifact
            2. Split into X / y
            3. Train XGBoost (scale_pos_weight computed from y_train)
            4. Evaluate on test set — recall is the primary metric
            5. Reject model if recall < expected_accuracy threshold
            6. Wrap model + preprocessor into USvisaModel and save
            7. Return ModelTrainerArtifact
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # ── 1. Load arrays ─────────────────────────────────────────
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            logging.info(f"Loaded train array {train_arr.shape} | test array {test_arr.shape}")

            # ── 2. Split X / y ─────────────────────────────────────────
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test,  y_test  = test_arr[:,  :-1], test_arr[:,  -1]

            logging.info(f"x_train: {x_train.shape} | y_train: {y_train.shape}")
            logging.info(f"x_test : {x_test.shape}  | y_test : {y_test.shape}")

            # ── 3. Train XGBoost ───────────────────────────────────────
            trained_model = self.train_xgboost(x_train, y_train)

            # ── 4. Evaluate ────────────────────────────────────────────
            metric_artifact, recall = self.evaluate_model(trained_model, x_test, y_test)

            # ── 5. Recall threshold check ──────────────────────────────
            # expected_accuracy in config acts as minimum acceptable recall.
            # The notebook achieved ~88.5% recall (threshold 0.5) on the test set.
            if recall < self.model_trainer_config.expected_accuracy:
                logging.info(
                    f"Recall {round(recall, 4)} is below the required threshold "
                    f"{self.model_trainer_config.expected_accuracy}. Model rejected."
                )
                raise Exception(
                    f"Model recall ({round(recall, 4)}) did not meet the "
                    f"expected threshold ({self.model_trainer_config.expected_accuracy}). "
                    f"Re-check data or hyperparameters."
                )

            logging.info(
                f"Model accepted — recall {round(recall, 4)} >= "
                f"threshold {self.model_trainer_config.expected_accuracy}"
            )

            # ── 6. Load preprocessor and build USvisaModel ─────────────
            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model,
            )
            logging.info("Created USvisaModel with preprocessor and XGBoost model")

            # Create output directory and save
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)
            logging.info(
                f"Saved USvisaModel to: {self.model_trainer_config.trained_model_file_path}"
            )

            # ── 7. Return artifact ─────────────────────────────────────
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e