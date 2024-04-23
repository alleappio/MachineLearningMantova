import warnings
import pandas as pd

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from typing import List

warnings.simplefilter(action='ignore', category=FutureWarning)


def distplot(df, df_norm, feature, color='g'):

    figure, axes = plt.subplots(1, 2, figsize=(8, 6))
    figure.suptitle("Distribution for {}".format(feature))

    axes[0].hist(df[feature])
    axes[1].hist(df_norm[feature], color='g')
    axes[0].set_title('No normalization applied.')
    axes[1].set_title('Standard scaling.')

    figure.show()


def cat_to_discrete(df: pd.DataFrame, categorical_cols: List[str]):

    X_cat = df[categorical_cols].copy()

    # passaggio categorico -> discreto:
    for col in categorical_cols:
        X_cat[col], _ = pd.factorize(X_cat[col], sort=True)

    return X_cat


class PreProcessing:

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    def __init__(self, target_col: str = 'Churn',
                 output_dictionary: dict = {'Yes': 0, 'No': 1},
                 scaler: StandardScaler = None, verbose: bool = True,
                 do_plot: bool = False):

        self.target_col = target_col
        self.output_dictionary = output_dictionary
        self.scaler = scaler
        self.verbose = verbose
        self.do_plot = do_plot

    def split_into_features_and_target(self, df: pd.DataFrame):

        # seleziona le features e la variabile target
        # passaggio categorico -> discreto
        y = df[self.target_col]

        if self.output_dictionary is not None:
            y = y.replace(self.output_dictionary)

        X = df.drop(columns=self.target_col)

        return X, y

    def check_null_values(self, X: pd.DataFrame):

        if self.verbose:
            # valori mancanti
            print('Valori nan: \n', X.isnull().any())
            print('Valori mancanti: \n', (X == ' ').any())

        X['TotalCharges'] = X['TotalCharges'].replace(" ", 0).astype('float32')
        X.drop(['customerID'], axis=1, inplace=True)

        return X

    def get_scaler(self):
        return self.scaler

    def spit_into_cat_and_num(self, X: pd.DataFrame):

        if self.verbose:
            # features categoriche
            print(X.info())  # le feature categoriche sono quelle di tipo non numerico.

        # NB: Anche SeniorCitizen è di fatti categorica, dato che assume solo due valori
        # corrispondenti a vero (1) o falso (0)
        categorical_cols = [c for c in X.columns if X[c].dtype == 'object'
                            or c == 'SeniorCitizen']

        X_cat = cat_to_discrete(X, categorical_cols)

        X_num = X[self.numerical_cols].astype('float64')

        if self.verbose:
            X_num.describe()

        if self.scaler is None:
            self.scaler = StandardScaler().fit(X_num)

        X_num = self.scaler.transform(X_num)
        X_num = pd.DataFrame(X_num, columns=self.numerical_cols)

        if self.verbose and self.do_plot:

            for feat in self.numerical_cols:
                distplot(X, X_num, feat)

        return (X_num, X_cat)


    def __call__(self, df: pd.DataFrame):
        X, y = self.split_into_features_and_target(df)
        X = self.check_null_values(X)
        X_num, X_cat = self.spit_into_cat_and_num(X)
        return (X_num, X_cat), y


