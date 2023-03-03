"""
    Module to test churn_library.py
"""
from datetime import datetime
import logging
from pathlib import Path

from churn_library import ChurnPredict
from configuration import conf


cpTestClass = ChurnPredict("./data/bank_data.csv")
cpTestClass.get_logger(
    f"test_churn_predict_{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}.log")


def test_import():
    """
        test import_data
    """
    cpTestClass.log.info("Testing import_data")
    try:
        df = cpTestClass.import_data()
        cpTestClass.log.info("TEST: Run import_data: SUCCESS")
    except FileNotFoundError as err:
        cpTestClass.log.error(
            "TEST: Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        cpTestClass.log.error(
            "TEST: Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    """
        test perform_eda function
    """
    logging.info("TEST: Testing perform_eda")
    try:
        cpTestClass.df["Churn"] = cpTestClass.df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        cpTestClass.perform_eda()
        logging.info("TEST: Run perform_eda: SUCCESS")
    except Exception as err:
        logging.error("TEST: Testing perform_eda: FAILED")
        raise err

    try:
        assert Path(cpTestClass.outputs_dir / "eda/churn_amount.png").exists()
        assert Path(cpTestClass.outputs_dir / "eda/customer_age.png").exists()
        assert Path(
            cpTestClass.outputs_dir /
            "eda/marital_status.png").exists()
        assert Path(cpTestClass.outputs_dir /
                    "eda/transactions_distribution.png").exists()
        assert Path(cpTestClass.outputs_dir / "eda/heatmap.png").exists()
    except AssertionError as err:
        logging.error(
            "TEST: Testing perform_eda: One or more of the expected outputs were not found")
        raise err


def test_encoder_helper():
    """
        test encoder_helper
    """
    cpTestClass.log.info("TEST: Testing encoder_helper")
    try:
        cpTestClass.encoder_helper(conf["category_columns"])
        cpTestClass.log.info("TEST: Run encoder_helper: SUCCESS")
    except Exception as err:
        cpTestClass.log.error("TEST: Testing encoder_helper: FAILED")
        raise err

    try:
        for category in conf["category_columns"]:
            assert f"{category}_Churn" in cpTestClass.df
    except AssertionError as err:
        cpTestClass.log.error(
            "TEST: Testing encoder_helper: One or more of the expected columns were not found")
        raise err


def test_perform_feature_engineering():
    """
        test perform_feature_engineering
    """
    cpTestClass.log.info("TEST: Testing perform_feature_engineering")
    try:
        cpTestClass.x_train, cpTestClass.x_test, cpTestClass.y_train, cpTestClass.y_test = cpTestClass.perform_feature_engineering()
        cpTestClass.log.info("TEST: Run perform_feature_engineering: SUCCESS")
    except Exception as err:
        cpTestClass.log.error(
            "TEST: Testing perform_feature_engineering: FAILED")
        raise err

    try:
        for col in conf["keep_cols"]:
            assert col in cpTestClass.df
    except AssertionError as err:
        cpTestClass.log.error(
            "TEST: Testing perform_feature_engineering: One or more of the expected columns were not found")
        raise err

    try:
        assert cpTestClass.x_train.shape[0] > 0
        assert cpTestClass.x_train.shape[1] > 0
        assert cpTestClass.x_test.shape[0] > 0
        assert cpTestClass.x_test.shape[1] > 0
        assert cpTestClass.y_train.shape[0] > 0
        assert cpTestClass.y_test.shape[0] > 0
    except AssertionError as err:
        cpTestClass.log.error(
            "TEST: Testing perform_feature_engineering: One or more data output are empty")
        raise err


def test_train_models():
    """
        test train_models
    """
    cpTestClass.log.info("TEST: Testing train_models")
    try:
        cpTestClass.train_models()
        cpTestClass.log.info("TEST: Run train_models: SUCCESS")
    except Exception as err:
        cpTestClass.log.error("TEST: Testing train_models: FAILED")
        raise err

    try:
        assert Path(cpTestClass.outputs_dir /
                    "results/feature_importance.png").exists()
        assert Path(
            cpTestClass.outputs_dir /
            "results/logistic_regression_classification_results.csv").exists()
        assert Path(
            cpTestClass.outputs_dir /
            "results/random_forest_classification_results.csv").exists()
        assert Path(
            cpTestClass.outputs_dir /
            "results/roc_curves.png").exists()
        assert Path(
            cpTestClass.outputs_dir /
            "models/logistic_model.pkl").exists()
        assert Path(cpTestClass.outputs_dir / "models/rfc_model.pkl").exists()
    except AssertionError as err:
        logging.error(
            "TEST: Testing train_models: One or more of the expected outputs were not found")
        raise err
