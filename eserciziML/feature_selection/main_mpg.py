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
    parser.add_argument("--verbose", type=str2bool, default=True)

    args = parser.parse_args()

    return args


def fill_na(df: pd.DataFrame, column_label: str = 'HP',
            verbose: bool = False):

    if verbose:
        print("\nChecking for NaN/null values.")
        print(df.isnull().any())

    ## TODO: fill the null values using the average value
    df = ...

    return df


class CarMPGPredictor:

    def __init__(self):
        # TODO: Scalamento delle feature usando la standardizzazione
        self.preprocessing = None

        # TODO: istanzio la Ridge regression
        self.regressor = None


    def fit(self, X_train, y_train):

        # T0DO: calcolo i parametri di trasformazione
        pass

        # T0DO: applico la transformazione
        pass

        # T0DO: fitto il regressore
        pass

    def predict(self, X):

        # T0DO: applico la trasformazione
        pass

        # T0DO: effettuo la predizione
        pass

    def get_regressor_coeffs(self):
        return self.regressor.coef_

    def print_metrics(self, y_val: np.array, y_pred: np.array,
                      do_plot: bool = True):

        print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))

        if do_plot:
            # TODO: realizzare un plot di visualizzazione che mostri le predizioni e gli errori del modello
            pass


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

    # TODO: modificare la funzione fill_na per gestire i valori nulli
    # Gestistico i valori nulli
    fill_na(df, verbose=args.verbose)

    # TODO: Sulla base della correlation matrix, scegliere le 4 features "migliori"
    # Plottare la correlation matrix, calcolata con pandas
    plot_corr_matrix(df, verbose=args.verbose)
    features = [..., ..., ..., ...]

    X, y = df[features].values, df['MPG'].values

    # TODO: usare la funzione train_test_split per dividere il dataset in train e val
    X_train, X_val, y_train, y_val = None

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

    # TODO: Misuro i risultati sul validation set usando MAE, MSE ed RMSE
    pass

    ########### Testing sul testing set ######################

    # TODO: effettuare il test
    # 1) caricare il dataset (path in args.dataset_path_test)
    # 2) gestire i valori nulli
    # 3) selezionare le colonne giuste su cui fare il test
    # 4) effettuare la predizione
    # 5) stampare le metriche