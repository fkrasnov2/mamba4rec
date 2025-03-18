import argparse

from mamba4rec import Datasets, TrainModel, Vocab, s3_tools


class Pipeline(s3_tools):
    def __init__(self, **creds):
        super().__init__(**creds)

    def load(self, bucket_name, data_key_name) -> dict:
        return self.get_dill_object(bucket_name=bucket_name, key_name=data_key_name)

    def train(self, data_dict: dict):
        vocab = Vocab(data_dict.get("search_texts", {}))
        datasets = Datasets(
            data_dict.get("train_interactions", []),
            data_dict.get("test_interactions", []),
        )
        model_trainer = TrainModel(vocab, datasets)

        model_trainer.generate()
        model_trainer.ndcg()
        model_trainer.save("./saved")

        print(model_trainer._metrics)

    def save(self, bucket_name, model_folder_name):
        self.safe_upload_folder(
            folder_name="./saved/*",
            bucket_name=bucket_name,
            object_name=model_folder_name.strip("/") + "/",
        )

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-bn",
            "--bucket_name",
            type=str,
            required=True,
            help="Bucket S3 dataset",
        )
        parser.add_argument(
            "-dkn",
            "--data_key_name",
            type=str,
            required=True,
            help="Path to S3 object",
        )
        parser.add_argument(
            "-mfn",
            "--model_folder_name",
            type=str,
            required=True,
            help="Path to S3 model folder",
        )
        args = parser.parse_args()
        return args


if __name__ == "__main__":

    pipeline = Pipeline()
    args = pipeline.parse()
    print(vars(args), flush=True)
    data_dict = pipeline.load(args.bucket_name, args.data_key_name)
    pipeline.train(data_dict)
    pipeline.save(args.bucket_name, args.model_folder_name)
