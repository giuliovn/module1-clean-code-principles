"""
Library for churn prediction
Author: Giulio Venturi
Date: March 2023
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

from configuration import conf

os.environ["QT_QPA_PLATFORM"] = "offscreen"
rcParams.update({"figure.autolayout": True})
sns.set()


class ChurnPredict:
    """
    Class to collect methods for churn prediction
    """

    def __init__(self, data_path, outputs_dir=Path("outputs")):
        self.outputs_dir = outputs_dir
        self.log = self.get_logger()
        self.data_path = data_path
        self.df = self.import_data()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def predict(self):
        self.log.debug("Add Churn column to dataframe")
        self.df["Churn"] = self.df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        self.perform_eda()
        self.encoder_helper(conf["category_columns"])
        self.x_train, self.x_test, self.y_train, self.y_test = self.perform_feature_engineering()
        self.train_models()

    def get_logger(
            self,
            log_file_name=f"churn_predict_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log",
            log_level=logging.INFO):
        logs_dir = self.outputs_dir / "logs"
        self.make_dir(logs_dir)
        log_path = logs_dir / log_file_name
        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()
        log = logging.getLogger()
        log.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s\t[%(levelname)s]\t%(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        log.addHandler(stream_handler)
        log.info(f"Log to {log_path}")
        return log

    def import_data(self):
        """
        returns dataframe for the csv found at pth

        input:
                data_path: a path to the csv
        output:
                df: pandas dataframe
        """
        self.log.info(f"Reading {self.data_path}")
        df = pd.read_csv(self.data_path)
        self.log.info(df.columns)
        return df

    def perform_eda(self):
        """
        perform eda on df and save figures to images folder
        """
        self.log.info("Perform exploratory analysis")
        images_output = self.outputs_dir / "eda"
        self.make_dir(images_output)

        plt.figure(figsize=(5, 5))
        churn_fig = self.df["Churn"].apply(
            lambda val: "No Churn" if val == 0 else "Churn").hist(
            bins=2).get_figure()
        churn_fig_path = images_output / "churn_amount.png"
        self.log.info(f"Churn amount visualization to {churn_fig_path}")
        churn_fig.savefig(churn_fig_path)

        plt.figure(figsize=(20, 10))
        age_fig = self.df["Customer_Age"].hist().get_figure()
        age_fig_path = images_output / "customer_age.png"
        self.log.info(f"Age visualization to {age_fig_path}")
        age_fig.savefig(age_fig_path)

        plt.figure(figsize=(20, 10))
        mar_status_fig = self.df.Marital_Status.value_counts(
            "normalize").plot(kind="bar").get_figure()
        mar_status_fig_path = images_output / "marital_status.png"
        self.log.info(f"Marital status visualization to {mar_status_fig_path}")
        mar_status_fig.savefig(mar_status_fig_path)

        plt.figure(figsize=(20, 10))
        trans_distr_fig = sns.histplot(
            self.df["Total_Trans_Ct"],
            stat="density",
            kde=True).get_figure()
        trans_distr_fig_path = images_output / "transactions_distribution.png"
        self.log.info(
            f"Transactions distribution visualization to {trans_distr_fig_path}")
        trans_distr_fig.savefig(trans_distr_fig_path)

        plt.figure(figsize=(20, 10))
        heatmap_fig = sns.heatmap(
            self.df.corr(),
            annot=False,
            cmap="Dark2_r",
            linewidths=2).get_figure()
        heatmap_fig_path = images_output / "heatmap.png"
        self.log.info(
            f"Heatmap correlation visualization to {heatmap_fig_path}")
        heatmap_fig.savefig(heatmap_fig_path)

    def encoder_helper(self, category_lst):
        """
        helper function to turn each categorical column into a new column with
        proportion of churn for each category

        input:
                category_lst: list of columns that contain categorical features
        """
        self.log.info(
            f"Add column for correlation with churn for each category: {', '.join(category_lst)}")
        for category in category_lst:
            category_groups = self.df.groupby(category).mean()["Churn"]
            self.df[f"{category}_Churn"] = self.df[category].apply(
                lambda val: category_groups[val])

    def perform_feature_engineering(self):
        """
        input:
                  df: pandas dataframe

        output:
                  x_train: X training data
                  x_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        """
        self.log.debug("Clean dataframe columns")

        churn = self.df["Churn"]
        clean_df = self.df[self.df.columns.intersection(conf["keep_cols"])]
        self.df = clean_df

        self.log.info("Prepare train and test data")
        x_train, x_test, y_train, y_test = train_test_split(
            self.df, churn, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    def train_models(self):
        """
        train, store model results: images + scores, and store models
        input:
                  None
        output:
                  None
        """
        self.log.info("Start training")
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default "lbfgs" fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"]
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.x_train, self.y_train)

        lrc.fit(self.x_train, self.y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.x_test)

        y_train_preds_lr = lrc.predict(self.x_train)
        y_test_preds_lr = lrc.predict(self.x_test)

        models_output = self.outputs_dir / "models"
        self.make_dir(models_output)
        self.log.info(f"Save models to {models_output}")
        joblib.dump(cv_rfc.best_estimator_, models_output / "rfc_model.pkl")
        joblib.dump(lrc, models_output / "logistic_model.pkl")

        results_output = self.outputs_dir / "results"
        self.make_dir(results_output)
        self.roc_curves_plot(
            lrc,
            cv_rfc.best_estimator_,
            self.x_test,
            self.y_test,
            results_output)
        self.classification_report_image(
            self.y_train,
            self.y_test,
            y_train_preds_rf,
            y_test_preds_rf,
            "Random Forest",
            results_output)
        self.classification_report_image(
            self.y_train,
            self.y_test,
            y_train_preds_lr,
            y_test_preds_lr,
            "Logistic Regression",
            results_output)
        self.feature_importance_plot(
            cv_rfc.best_estimator_, self.df, results_output)

    def roc_curves_plot(self, lr_model, rfc_model, x_test, y_test, output_pth):
        """
        produces roc curves plot for the two models in output_pth
        input:
                lr_model: logistic regression model
                rfc_model: random forest regression model
                x_test: X testing data
                y_test: y testing data
                output_pth: path to store the figure

        output:
                 None
        """
        plt.figure(figsize=(15, 8))
        plot_roc_curve(lr_model, x_test, y_test)
        ax = plt.gca()
        plot_roc_curve(rfc_model, x_test, y_test, ax=ax, alpha=0.8)

        roc_fig_path = output_pth / "roc_curves.png"
        self.log.info(f"ROC plot to {roc_fig_path}")
        plt.savefig(roc_fig_path)

    def classification_report_image(self, y_train,
                                    y_test,
                                    y_train_preds,
                                    y_test_preds,
                                    model_type_str,
                                    output_pth):
        """
        produces classification report for training and testing results and stores report as image
        in output_pth
        input:
                y_train: training response values
                y_test: test response values
                y_train_preds: training predictions for a specific model
                y_test_preds: test predictions for a specific model
                model_type_str: name of the model type
                output_pth: path to store the figure

        output:
                 None
        """
        self.log.info(f"{model_type_str} results")
        self.log.info("Test results")
        test_results = classification_report(y_test, y_test_preds)
        self.log.info(test_results)
        self.log.info("Train results")
        train_results = classification_report(y_train, y_train_preds)
        self.log.info(train_results)

        csv_path = output_pth / \
            f"{model_type_str.replace(' ', '_').lower()}_classification_results.csv"
        self.log.info(f"Save {model_type_str.lower()} results to {csv_path}")
        with open(csv_path, "w+") as csv:
            csv.write("Test results\n")
            csv.write(test_results)

        with open(csv_path, "a") as csv:
            csv.write("\nTrain results\n")
            csv.write(train_results)

    @staticmethod
    def make_dir(dir_path):
        """
        makes sure a directory exists
        input:
                dir_path:
        output:
                None
        """
        dir_path.mkdir(parents=True, exist_ok=True)

    def feature_importance_plot(self, model, x_data,
                                output_pth):
        """
        creates and stores the feature importance in output_pth
        input:
                model: model object containing feature_importances_
                x_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                 None
        """
        # Calculate feature importances
        self.log.info("Calculate model feature importances")

        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importance
        names = [x_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel("Importance")

        # Add bars
        plt.bar(range(x_data.shape[1]), importances[indices])

        # Add feature names as x_data-axis labels
        plt.xticks(range(x_data.shape[1]), names, rotation=90)

        feature_importance_fig_path = output_pth / "feature_importance.png"
        self.log.info(
            f"Feature importance visualization to {feature_importance_fig_path}")
        plt.savefig(feature_importance_fig_path)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", help="Path to input data in csv format")
    parser.add_argument(
        "-o",
        "--output",
        help="Output base directory",
        default=Path("outputs"))
    args = parser.parse_args()
    ChurnPredict(args.data_path, ).predict()
