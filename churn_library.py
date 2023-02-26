"""
Library for churn prediction
Author: Giulio
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split

os.environ["QT_QPA_PLATFORM"] = "offscreen"

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format="%(asctime)s\t[%(levelname)s]\t%(message)s",
)
log = logging.getLogger()


class ChurnPredict:
    def __init__(self, data_path):
        self.df = self.import_data(data_path)
        log.debug(f"Add Churn column to dataframe")
        self.df["Churn"] = self.df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
        self.perform_eda()
        self.perform_feature_engineering()

    def import_data(self, data_path):
        """
        returns dataframe for the csv found at pth

        input:
                data_path: a path to the csv
        output:
                df: pandas dataframe
        """
        log.info(f"Reading {data_path}")
        df = pd.read_csv(data_path)
        log.info(df.columns)
        return df

    def perform_eda(self, images_output=Path("images/eda")):
        """
        perform eda on df and save figures to images folder
        """
        log.debug(f"Make sure images output directory {images_output} exists")
        images_output.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(5, 5))
        churn_fig = self.df["Churn"].apply(lambda val: "No Churn" if val == 0 else "Churn").hist(bins=2).get_figure()
        churn_fig_path = images_output / "churn_amount.png"
        log.info(f"Churn amount visualization to {churn_fig_path}")
        churn_fig.savefig(churn_fig_path)

        plt.figure(figsize=(20, 10))
        age_fig = self.df["Customer_Age"].hist().get_figure()
        age_fig_path = images_output / "customer_age.png"
        log.info(f"Age visualization to {age_fig_path}")
        age_fig.savefig(age_fig_path)

        plt.figure(figsize=(20, 10))
        mar_status_fig = self.df.Marital_Status.value_counts("normalize").plot(kind="bar").get_figure()
        mar_status_fig_path = images_output / "marital_status.png"
        log.info(f"Marital status visualization to {mar_status_fig_path}")
        mar_status_fig.savefig(mar_status_fig_path)

        plt.figure(figsize=(20, 10))
        trans_distr_fig = sns.histplot(self.df["Total_Trans_Ct"], stat="density", kde=True).get_figure()
        trans_distr_fig_path = images_output / "transactions_distribution.png"
        log.info(f"Transactions distribution visualization to {trans_distr_fig_path}")
        trans_distr_fig.savefig(trans_distr_fig_path)

        plt.figure(figsize=(20, 10))
        heatmap_fig = sns.heatmap(self.df.corr(), annot=False, cmap="Dark2_r", linewidths=2).get_figure()
        heatmap_fig_path = images_output / "heatmap.png"
        log.info(f"Heatmap correlation visualization to {heatmap_fig_path}")
        heatmap_fig.savefig(heatmap_fig_path)

    def encoder_helper(self, category_lst):
        """
        helper function to turn each categorical column into a new column with
        proportion of churn for each category

        input:
                category_lst: list of columns that contain categorical features
        """
        log.info(f"Add column for correlation with churn for each category: {', '.join(category_lst)}")
        for category in category_lst:
            category_groups = self.df.groupby(category).mean()["Churn"]
            self.df[f"{category}_Churn"] = self.df[category].apply(lambda val: category_groups[val])

    def perform_feature_engineering(self):
        """
        input:
                  df: pandas dataframe

        output:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        """
        category_columns = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"
        ]
        self.encoder_helper(category_columns)

        log.debug("Clean dataframe columns")
        keep_cols = ["Churn", "Customer_Age", "Dependent_count", "Months_on_book",
                     "Total_Relationship_Count", "Months_Inactive_12_mon",
                     "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
                     "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
                     "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
                     "Gender_Churn", "Education_Level_Churn", "Marital_Status_Churn",
                     "Income_Category_Churn", "Card_Category_Churn"]

        clean_df = self.df[self.df.columns.intersection(keep_cols)]
        self.df = clean_df

        log.info("Prepare train and test data")
        x_train, x_test, y_train, y_test = train_test_split(self.df, self.df["Churn"], test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", help="Path to input data in csv format")
    args = parser.parse_args()
    log.debug(args)
    ChurnPredict(args.data_path)
