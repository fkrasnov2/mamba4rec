import os
import time
from glob import glob

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import dill


class s3_tools:
    def __init__(self, **creds):
        config = Config(signature_version="s3v4")
        self._s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_URL", creds.get("S3_URL")),
            aws_access_key_id=os.environ.get(
                "S3_ACCESS_KEY", creds.get("S3_ACCESS_KEY")
            ),
            aws_secret_access_key=os.environ.get("S3_SECRET", creds.get("S3_SECRET")),
            aws_session_token="session_token",
            config=config,
            region_name="us-east-1",
            verify=True,
        )

    @property
    def s3_client(self) -> boto3.client:
        return self._s3_client

    def upload_file(self, file_name, bucket_name, object_name=None) -> bool:
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """

        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = os.path.basename(file_name)

        # Upload the file
        try:
            self.s3_client.upload_file(file_name, bucket_name, object_name)
        except ClientError:
            return False
        return True

    def check_exists(self, bucket_name: str, key_name: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=key_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # The key does not exist.
                return False
        return True

    def safe_upload_folder(
        self, folder_name: str, bucket_name: str, object_name=None
    ) -> bool:
        if object_name is None:
            object_name = folder_name

        if self.check_exists(bucket_name, object_name):
            object_name = object_name.strip("/") + str(int(time.perf_counter())) + "/"
        print(f"{object_name=}")
        # self.s3_client.put_object(Bucket=bucket_name, Key=object_name)
        results = []
        for file_name in glob(folder_name):
            res = self.upload_file(
                file_name,
                bucket_name,
                object_name=object_name + os.path.basename(file_name),
            )
            results.append(res)
            print(os.path.basename(file_name), res)
        return all(results)

    def download_folder(self, bucket_name: str, object_name: str, folder_name: str):
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        for key in self.s3_client.list_objects(Bucket=bucket_name)["Contents"]:
            if key["Key"].startswith(object_name):
                print(key["Key"])
                self.s3_client.download_file(
                    Bucket=bucket_name,
                    Key=key["Key"],
                    Filename=folder_name + "/" + os.path.basename(key["Key"]),
                )

    def get_dill_object(self, bucket_name: str, key_name: str) -> dict:
        body = self.s3_client.get_object(Bucket=bucket_name, Key=key_name)
        return dill.loads(body["Body"].read())
