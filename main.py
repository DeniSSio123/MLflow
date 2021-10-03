import argparse
import mlflow

import urllib.request
import zipfile
import pandas as pd
import numpy as np

from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import mlflow.xgboost


def parse_args():
    parser = argparse.ArgumentParser(description="URL")
    parser.add_argument(
        "--url",
        type=str,
        default='http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        help="url",
    )
    parser.add_argument(
        "--max_row_limit",
        type=int,
        default=10000,
        help="parquet file row number",
    )
    parser.add_argument(
        "--split_prop",
        type=float,
        default=0.8,
        help="train split percent",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="algoritm iteration count",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="parquet file row number",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=12,
        help="matrix rang",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.3,
        help="eta xgb",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=6,
        help="max deth xgboost",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=10,
        help="xgb num rounds",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    # parse command-line arguments
    url = args.url
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    zip_file_object.extractall()



    with mlflow.start_run():

        mlflow.log_artifact('/content/ml-latest-small/ratings.csv')
        artifact_uri_df = mlflow.get_artifact_uri()[7:] + '/ratings.csv'



    df = pd.read_csv(artifact_uri_df)
    df = df.drop(columns = ['timestamp'])
    df.iloc[:args.max_row_limit].to_parquet('df.parquet.gzip',
                  compression='gzip')



    with mlflow.start_run():

        mlflow.log_artifact('/content/df.parquet.gzip')
        artifact_uri_parq = mlflow.get_artifact_uri()[7:] + '/df.parquet.gzip'

    df = pd.read_parquet(artifact_uri_parq)

    X_train, X_test= train_test_split(df, train_size = args.split_prop, random_state = 0)

    X_train = pd.pivot_table(X_train, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.mean).fillna(0)
    X_test = pd.pivot_table(X_test, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.mean).fillna(0)

    X_test_fixed = X_train.copy()
    X_test_fixed[:] = 0
    X_test_fixed[[k for k in X_test.columns if k in X_train.columns]] = X_test[[k for k in X_test.columns if k in X_train.columns]]


    with mlflow.start_run():
      estimator = NMF(n_components=args.n_components, max_iter = args.max_iter, alpha = args.alpha, random_state=0)
      estimator.fit(X_train)
      H = estimator.components_
      mlflow.log_metric("Accuracy_test", mean_squared_error(X_test_fixed, np.dot(estimator.transform(X_test_fixed), H)))
      mlflow.log_metric("Accuracy_train", mean_squared_error(X_train, np.dot(estimator.transform(X_train), H)))
      mlflow.log_param("test_value_count", X_test_fixed.shape[0])
      mlflow.log_param("train_value_count", X_train.shape[0])
      mlflow.sklearn.log_model(estimator, artifact_path="models")
      artifact_uri_model = mlflow.get_artifact_uri(artifact_path="models")[7:]

    pymodel = mlflow.sklearn.load_model(artifact_uri_model)

    df = pd.read_parquet(artifact_uri_parq)

    X_train_orig, X_test_orig= train_test_split(df, train_size = 0.8, random_state = 0)

    X_train = pd.pivot_table(X_train_orig, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.mean).fillna(0)
    X_test = pd.pivot_table(X_test_orig, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.mean).fillna(0)

    X_test_fixed = X_train.copy()
    X_test_fixed[:] = 0
    X_test_fixed[[k for k in X_test.columns if k in X_train.columns]] = X_test[[k for k in X_test.columns if k in X_train.columns]]

    H = pymodel.components_
    W = pymodel.transform(X_train)

    df_W = pd.DataFrame(W).set_index(pd.DataFrame(W).index + 1)
    df_H = pd.DataFrame(H).T.set_index(X_train.columns)

    X_train_xgb = X_train_orig.merge(df_W, left_on = 'userId', right_index = True).merge(df_H, left_on = 'movieId', right_index = True)

    W = pymodel.transform(X_test_fixed)
    df_W = pd.DataFrame(W).set_index(pd.DataFrame(W).index + 1)

    X_test_xgb = X_test_orig.merge(df_W, left_on = 'userId', right_index = True).merge(df_H, left_on = 'movieId', right_index = True)


    mlflow.xgboost.autolog()


    with mlflow.start_run():
      #estimator = NMF(n_components=args.n_components, max_iter = args.max_iter, alpha = args.alpha, random_state=0)
      #estimator.fit(X_train)
      H = pymodel.components_
      dtrain = xgb.DMatrix(X_train_xgb.drop(columns = ['userId', 'movieId', 'rating']), X_train_xgb.rating)
      dtest = xgb.DMatrix(X_test_xgb.drop(columns = ['userId', 'movieId', 'rating']), X_test_xgb.rating)
      
      params = {
            "objective": "reg:squarederror",
            "max_depth ": args.max_depth ,
            "eta": args.eta,
            "seed": 42,
        }
      model = xgb.train(params, dtrain, args.num_rounds)

      # evaluate model
      y_proba = model.predict(dtest)
      mse = mean_squared_error(X_test_xgb.rating, y_proba)

      # log metrics
      mlflow.log_metrics({"MSE": mse})



if __name__ == "__main__":
    main()

