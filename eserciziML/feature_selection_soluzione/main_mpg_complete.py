import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn import metrics

from utils import str2bool
from utils import print_info
from utils import plot_corr_matrix


def read_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path_train', type=str, default='datasets/auto-MPG/auto-MPG_tr.csv',
                        help='Path to the file containing the training set.')
    parser.add_argument('--dataset_path_test', type=str, default='datasets/auto-MPG/auto-MPG_te.csv',
                        help='Path to the file containing the test set.')
    parser.add_argument('--val_percent', type=float, default=0.2,
                        help='Percentage of elements that will be used for the validation set.')
    parser.add_argument("--verbose", type=str2bool, default=False)

    args = parser.parse_args()

    return args


def fill_na(df: pd.DataFrame, column_label: str = 'HP',
            verbose: bool = False):

    if verbose:
        print("\nChecking for NaN/null values.")
        print(df.isnull().any())

    selcol = df[column_label]
    m = selcol.mean()
    df[column_label] = selcol.fillna(m)

    return df

class CarMPGPredictor:

    def __init__(self):
        self.preprocessing = StandardScaler()

        self.regressor = Ridge(alpha=0.5)


    def fit(self, X_train, y_train):

        # calcolo i parametri di trasformazione
        self.preprocessing.fit(X_train)

        # applico la transformazione
        X_train_transf = self.preprocessing.transform(X_train)

        self.regressor.fit(X_train_transf, y_train)

    def predict(self, X):
        X_transf = self.preprocessing.transform(X)
        return self.regressor.predict(X_transf)

    def get_regressor_coeffs(self):
        return self.regressor.coef_

    def print_metrics(self, y_val: np.array, y_pred: np.array,
                      do_plot: bool = True):

        print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))

        if do_plot:
            df = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred}).head(25)
            df.plot(kind='bar', figsize=(10, 8))
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            plt.show()


class CarMPGNonLinearPredictor(CarMPGPredictor):

    def __init__(self):
        super().__init__()
        self.regressor = LinearRegression()

    def feature_transformation(self, X):
        return np.asarray([X[:, 0], np.log(X[:, 3]), 1 / X[:, 1],
                           1 / X[:, 2]]).transpose()

    def fit(self, X_train, y_train):
        X_train_transf = self.feature_transformation(X_train)
        self.regressor.fit(X_train_transf, y_train)

    def predict(self, X):
        X_transf = self.feature_transformation(X)
        return self.regressor.predict(X_transf)


if __name__ == '__main__':

    args = read_args()

    # load dataset
    df = pd.read_csv(args.dataset_path_train)
    # more from: http://archive.ics.uci.edu/ml/datasets/Auto+MPG
    # MPG è la variabile target

    if args.verbose:
        print("\nGeneric information about the dataset.")
        # Un'occhiata al dataset
        print_info(df)

    # Gestistico i valori nulli
    fill_na(df, verbose=args.verbose)

    # Plottare la correlation matrix, calcolata con pandas
    plot_corr_matrix(df, verbose=args.verbose)

    features = ['CYL', 'DIS', 'HP', 'WGT']

    X, y = df[features].values, df['MPG'].values

    # divido il dataset in training e validation
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=args.val_percent, random_state=0)

    model = CarMPGPredictor()

    ########### Training ######################
    model.fit(X_train, y_train)

    if args.verbose:
        print("\nCoefficients learned during ridge regression.")
        coeff_df = pd.DataFrame(model.get_regressor_coeffs(),
                                features, columns = ['Coefficient'])
        print(coeff_df)

    ########### Testing su train e val set. ######################
    y_pred = model.predict(X_train)

    print("\nPerformance on the training set.")
    model.print_metrics(y_train, y_pred, do_plot=False)

    y_pred = model.predict(X_val)

    print("\nPerformance on the validation set.")
    model.print_metrics(y_val, y_pred, do_plot=False)

    # load dataset
    df_test = pd.read_csv(args.dataset_path_test)

    fill_na(df, verbose=args.verbose)

    X_test, y_test = df[features].values, df['MPG'].values

    ########## Testing sul testing set ######################
    y_pred = model.predict(X_test)

    print("\nPerformance on the test set.")
    model.print_metrics(y_test, y_pred, do_plot=False)

    #######################################################################################################################
    ## TODO: uso un modello (una "classe di modelli") predittivo non lineare trasformando non linearmente le feature o alcune di esse
    #
    model2 = CarMPGNonLinearPredictor()
    model2.fit(X_train, y_train)

    y_pred = model2.predict(X_train)
    print("\nPerformance on the training set.")
    model.print_metrics(y_train, y_pred, do_plot=False)

    y_pred = model2.predict(X_val)
    print("\nPerformance on the validation set.")
    model.print_metrics(y_val, y_pred, do_plot=False)

    y_pred = model.predict(X_test)
    print("\nPerformance on the test set.")
    model.print_metrics(y_test, y_pred, do_plot=False)
