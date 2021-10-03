import argparse
import mlflow

import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(description="URL")
    parser.add_argument(
        "--ratings_csv",
        type=str,
        help="csv file lcoation",
    )
    parser.add_argument(
        "--max_row_limit",
        type=int,
        default=10000,
        help="parquet file row number",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(args.ratings_csv)
    df = df.drop(columns = ['timestamp'])
    df.iloc[:args.max_row_limit].to_parquet('df.parquet.gzip',
                  compression='gzip')



    with mlflow.start_run():

        mlflow.log_artifact('/content/df.parquet.gzip')


if __name__ == "__main__":
    main()