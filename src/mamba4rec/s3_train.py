import argparse
import mamba4rec
from mamba4rec import TrainModel, Datasets
from s3_tools import s3_tools
from dill import dumps

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
    print(vars(args), flush=True)

    s3 = s3_tools()
    data_dict = s3.get_dill_object(
        bucket_name=args.bucket_name, key_name=args.data_key_name
    )
    vocab = mamba4rec.Vocab(data_dict.get("search_texts", {}))
    TrainModel(
        vocab,
        Datasets(
            data_dict.get("train_interactions", []),
            data_dict.get("test_interactions", []),
        ),
    )

    with open("./saved/vocab.obj", "wb") as fn:
        fn.write(dumps(vocab))

    s3.safe_upload_folder(
        folder_name="./saved/*",
        bucket_name=args.bucket_name,
        object_name=args.model_folder_name.strip("/") + "/",
    )
