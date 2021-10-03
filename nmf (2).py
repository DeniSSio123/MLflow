import argparse
import mlflow

import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def parse_args():
    parser = argparse.ArgumentParser(description="URL")
    parser.add_argument(
        "--ratings_data",
        type=str,
        help="parquet file lcoation",
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
    return parser.parse_args()

def main():
    args = parse_args()

    df = pd.read_parquet(args.ratings_data)

    X_train, X_test= train_test_split(df, train_size = args.split_prop, random_state = 0)

    X_train = pd.pivot_table(X_train, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.mean).fillna(0)
    X_test = pd.pivot_table(X_test, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.mean).fillna(0)

    #оставляем только нужные колонки, чтобы не падал трансформ на тесте
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



if __name__ == "__main__":
    main()