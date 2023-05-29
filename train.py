# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import warnings

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    precision = metrics.precision_score(actual, pred, average="weighted")
    recall = metrics.recall_score(actual, pred, average="weighted")
    f1score = metrics.f1_score(actual, pred, average="weighted")
    
    return accuracy, precision, recall,f1score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    
    
    df = pd.read_csv("pointure.data")
    
    label_encoder = preprocessing.LabelEncoder()
    input_classes = ['masculin','féminin']
    label_encoder.fit(input_classes)

# transformer un ensemble de classes
    encoded_labels = label_encoder.transform(df['Genre'])
    print(encoded_labels)
    df['Genre'] = encoded_labels
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(df)

    X = df.iloc[:, 1:4]
    y = df.iloc[:, 0]

    #decomposer les donnees predicteurs en training/testing
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=44)

    
    with mlflow.start_run():
        lr = GaussianNB()
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (accuracy, precision, recall,f1score) = eval_metrics(test_y, predicted_qualities)

        print("GaussianNB model: ")
        print("  accuracy: %s" % accuracy)
        print("  precision: %s" % precision)
        print("  recall: %s" % recall)
        print("  f1score: %s" % f1score)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1score", f1score)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        inputs = {'Taille(cm)':[183], 'Poids(kg)':[59], 'Pointure(cm)':[20]}
        dfToPredict = pd.DataFrame(data=inputs) 
        
        outputs = lr.predict(dfToPredict)
        print("La classe prédite est: ", outputs)
        if outputs == 0:
            genre = "Female"
        else:
            genre = "Male"
        mlflow.log_text("La classe prédite est: ", genre)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        #Pour la partie CML
        #Affichage des métriques
        # Write scores to a file
        with open("metrics.txt", 'w') as outfile:
            outfile.write("accuracy:  {0:2.1f} \n".format(accuracy))
            outfile.write("precision: {0:2.1f}\n".format(precision))
            outfile.write("recall:  {0:2.1f} \n".format(recall))
            outfile.write("f1score:  {0:2.1f} \n".format(f1score))

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="GaussianModelPointure", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)