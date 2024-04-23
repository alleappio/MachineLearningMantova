import numpy as np
import warnings
import pandas as pd
from typing import List

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, \
                            f1_score, precision_recall_curve, roc_curve, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB

import matplotlib.pyplot as plt

from utils import read_args
from utils import print_info
from preprocess import PreProcessing

warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_results(models_name: List[str], y: np.array,
                 y_predicted: List, y_confidences: List):

    for i in range(len(models_name)):
        print('/-------------------------------------------------------------------------------------------------------- /')
        print('RESULTS OF THE %s CLASSIFIER' % models_name[i])
        print('/-------------------------------------------------------------------------------------------------------- /')
        print('Accuracy is ', accuracy_score(y_test, y_predicted[i]))
        print('Precision is ', precision_score(y_test, y_predicted[i]))
        print('Recall is ', recall_score(y_test, y_predicted[i]))
        print('F1-Score is ', f1_score(y_test, y_predicted[i]))
        print('AUC is ', roc_auc_score(y_test, y_confidences[i]))

        fpr, tpr, _ = roc_curve(y_test, y_confidences[i])

        plt.figure('ROC')
        plt.title('ROC curve')
        plt.plot(fpr, tpr, label=models_name[i], linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        precision, recall, _ = precision_recall_curve(y_test, y_confidences[i])
        plt.figure('PR')
        plt.title('P-R curve')
        plt.plot(recall, precision, label=models_name[i], linewidth=4)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])

    plt.show()


def create_models():

    logr_model_plain = LogisticRegression()

    # Logistic Regression
    logr_model = LogisticRegression(penalty='l2', class_weight='balanced')

    # Naive Bayes
    nb_model = CategoricalNB(fit_prior=True, alpha=1.0)

    models_name = []
    models_name.append('Class-balanced Logistic Regression')
    models_name.append('Plain Logistic Regression')
    models_name.append('Naive Bayes')

    return (logr_model, logr_model_plain, nb_model), models_name


if __name__ == '__main__':

    args = read_args()

    df_train = pd.read_csv(args.dataset_path_train)

    print_info(df_train)

    pp_train = PreProcessing(target_col='Churn',
                             output_dictionary={'Yes': 0, 'No': 1},
                             verbose=args.verbose,
                             do_plot=True)

    (X_num_train, X_cat_train), y_train = pp_train(df_train)

    # features finali
    X_train = pd.concat([X_num_train, X_cat_train], axis=1, sort=False)

    # instanzia i modelli
    (logr_model, logr_model_plain, nb_model), \
        models_name = create_models()

    logr_model.fit(X_train, y_train)
    logr_model_plain.fit(X_train, y_train)
    # Attenzione: qui uso solo le feature categoriche!
    nb_model.fit(X_cat_train, y_train)

    df_test = pd.read_csv(args.dataset_path_test)

    pp_test = PreProcessing(target_col='Churn',
                            output_dictionary={'Yes': 0, 'No': 1},
                            scaler=pp_train.get_scaler(),
                            verbose=args.verbose)

    (X_num_test, X_cat_test), y_test = pp_test(df_test)

    # features finali
    X_test = pd.concat([X_num_test, X_cat_test], axis=1, sort=False)

    y_predicted, y_confidences = [], []

    # Predizione
    y_pred = logr_model.predict(X_test)
    y_predicted.append(y_pred)
    y_confidences.append(logr_model.predict_proba(X_test)[:, 1])

    y_pred = logr_model_plain.predict(X_test)
    y_predicted.append(y_pred)
    y_confidences.append(logr_model_plain.predict_proba(X_test)[:, 1])

    # Attenzione: qui uso solo le feature categoriche!
    y_pred = nb_model.predict(X_cat_test)
    y_predicted.append(y_pred)
    y_confidences.append(nb_model.predict_proba(X_cat_test)[:, 1])

    plot_results(models_name, y_test,
                 y_predicted, y_confidences)

