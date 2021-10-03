import argparse
import mlflow

import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import mlflow.xgboost

def parse_args():
    parser = argparse.ArgumentParser(description="URL")
    parser.add_argument(
        "--ratings_data",
        type=str,
        help="parquet file lcoation",
    )
    parser.add_argument(
        "--als_model_uri",
        type=str,
        help="model ALS",
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
    pymodel = mlflow.sklearn.load_model(args.als_model_uri)

    df = pd.read_parquet(args.ratings_data)

    X_train_orig, X_test_orig= train_test_split(df, train_size = 0.8, random_state = 0)

    X_train = pd.pivot_table(X_train_orig, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.mean).fillna(0)
    X_test = pd.pivot_table(X_test_orig, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.mean).fillna(0)

    X_test_fixed = X_train.copy()
    X_test_fixed[:] = 0
    X_test_fixed[[k for k in X_test.columns if k in X_train.columns]] = X_test[[k for k in X_test.columns if k in X_train.columns]]

    H = pymodel.components_
    W = pymodel.transform(X_train)
    
    #получаем фичи по юзерам и фильмам
    df_W = pd.DataFrame(W).set_index(pd.DataFrame(W).index + 1)
    df_H = pd.DataFrame(H).T.set_index(X_train.columns)

    #джойним фичи по юзерам и фильмам
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