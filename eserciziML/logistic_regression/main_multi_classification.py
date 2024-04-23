import warnings
import pandas as pd
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import read_args
from utils import print_info
from preprocess import PreProcessing


if __name__ == '__main__':

    pass
    # TODO
    # Partire dal codice di "esempio_binary_classification"
    # Modificarlo per effettuare una multi-classification

    # 1) Usiamo 'PaymentMethod' come variabile target. Tutte le altre sono le feature (con o senza 'Churn', a scelta)
    # 2) Come metriche, usiamo solo quelle che trovate in fondo
    # 3) Come modello usiamo la multinomial logistic regression, che in sklearn viene chiamata usando: "multi_class='multinomial'" (v. sotto))
