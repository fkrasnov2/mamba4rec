import argparse

import torch
from mamba4rec.s3_tools import s3_tools
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from mamba4rec.train import DataCollatorForCLMRec, ListDataset
from transformers import MambaForCausalLM
from transformers.generation.configuration_utils import GenerationConfig

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
        help="Path to S3 object model",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=True,
        help="batch size for inference",
        default=12,
    )

    args = parser.parse_args()
    print(vars(args), flush=True)

    s3 = s3_tools()
    data_dict = s3.get_dill_object(
        bucket_name=args.bucket_name, key_name=args.data_key_name
    )
    print(data_dict.keys())

    at_k = len(data_dict.get("test_interactions", [])[0])

    ##check if exist local folder or get S3
    s3.download_folder(
        bucket_name=args.bucket_name,
        object_name=args.model_folder_name,
        folder_name="./saved/",
    )
    model = MambaForCausalLM.from_pretrained("./saved/")
    pad_id = model.config.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    gconf = GenerationConfig(
        max_new_tokens=at_k,
        num_beams=1,
        do_sample=True,
        pad_token_id=pad_id,
        no_repeat_ngram_size=1,
    )

    train_inference = []
    train_dataloader = DataLoader(
        ListDataset(data_dict.get("train_interactions", [])),
        batch_size=args.batch_size,
        collate_fn=DataCollatorForCLMRec(pad_id),
        shuffle=False,
    )
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            train_inference += (
                model.generate(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    generation_config=gconf,
                )
                .detach()
                .cpu()[:, -at_k:]
                .tolist()
            )

    print(
        f'nDCG@{at_k} = {ndcg_score(data_dict.get("test_interactions", []), train_inference):.4f}'
    )
