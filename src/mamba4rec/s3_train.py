import argparse
import os

import boto3
import dill
from botocore.client import Config

from train import TrainModel, Dataloaders

def get_s3_client():
    config = Config(signature_version="s3v4")
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET"),
        aws_session_token="session_token",
        config=config,
        region_name="us-east-1",
        verify=False,
    )
    return s3_client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bn",
        "--bucket_name",
        type=str,
        required=True,
        help="Bucket S3 dataset",
    )
    parser.add_argument(
        "-kn",
        "--key_name",
        type=str,
        required=True,
        help="Path to S3 object",
    )

    args = parser.parse_args()
    print(vars(args), flush=True)

    s3_client = get_s3_client()
    body = s3_client.get_object(Bucket=args.bucket_name, Key=args.key_name)
    data_dict = dill.loads(body["Body"].read())
    print(data_dict.keys())
    tm = TrainModel(
        Dataloaders(
            data_dict.get("search_texts", set()),
            data_dict.get("train_interactions", []),
            data_dict.get("test_interactions", []),
        )
    )
