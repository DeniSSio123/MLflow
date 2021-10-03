import argparse
import mlflow

import urllib.request
import zipfile



def parse_args():
    parser = argparse.ArgumentParser(description="URL")
    parser.add_argument(
        "--url",
        type=str,
        default='http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        help="url",
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


if __name__ == "__main__":
    main()