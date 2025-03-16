import json

import dill
import numpy as np
import torch
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MambaConfig, MambaForCausalLM, Trainer, TrainingArguments
from transformers.generation.configuration_utils import GenerationConfig


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[list[int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __getitems__(self, idx_list):
        return [self.data[_] for _ in idx_list]

    def distinct_size(self) -> int:
        res = set()
        for lst in self.data:
            res.update(lst)
        return len(res)


class DataCollatorForCLMRec:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def mask_ids_batch(self, batch_of_ids):
        padded_ids = pad_sequence(
            list(map(lambda ids: torch.LongTensor(ids[::-1]), batch_of_ids)),
            batch_first=True,
            padding_value=self.pad_id,
        ).flip(dims=[1])
        return {
            "input_ids": padded_ids,
            "attention_mask": padded_ids != self.pad_id,
            "labels": padded_ids,
        }

    def __call__(self, batch_of_ids):
        return self.mask_ids_batch(batch_of_ids)


class Vocab:
    def __init__(self, vocab_raw: dict):
        """
        vocab_raw is item's data structure a dict of tuples (id, search_text_raw) with key search_text (which is normed form of search_text_raw)
        The goal of Vocab class is to extend vocab_raw with [PAD] and [UNK]
        """
        self._item2id = {item: raw_tuple[0] for item, raw_tuple in vocab_raw.items()}
        self._item2id[self.pad_str] = len(self._item2id)
        self._item2id[self.unk_str] = len(self._item2id)
        self._id2item = {idx: item for item, idx in self._item2id.items()}
        self._item2raw_item = {
            item: raw_tuple[1] for item, raw_tuple in vocab_raw.items()
        }

    def item_id_to_raw_item(self, item_id: int) -> str:
        return self.item2raw_item(self._id2item[item_id])

    def item2raw_item(self, item: str) -> str:
        return self._item2raw_item.get(item, self.unk_str)

    def __getitem__(self, idx: int) -> str:
        return self._id2item.get(idx)

    @property
    def vocab_size(self) -> int:
        return len(self._item2id)

    @property
    def id2item(self) -> dict:
        return self._id2item

    @property
    def item2id(self) -> dict:
        return self._item2id

    @property
    def pad_str(self) -> str:
        return "[PAD]"

    @property
    def pad_id(self) -> int:
        return self._item2id.get(self.pad_str, -1)

    @property
    def unk_str(self) -> str:
        return "[UNK]"

    @property
    def unk_id(self) -> int:
        return self._item2id.get(self.unk_str, -1)


class Datasets:
    def __init__(
        self, train_interactions: list[list[int]], test_interactions: list[list[int]]
    ):
        self._train_interactions = train_interactions
        self._test_interactions = test_interactions
        self._leave_k_out = len(test_interactions[0])

        (
            X_train,
            X_eval,
        ) = train_test_split(
            train_interactions,
            test_size=0.05,
            random_state=42,
        )

        self._train_dataset = ListDataset(X_train)
        self._eval_dataset = ListDataset(X_eval)

    @property
    def train_dataset(self) -> ListDataset:
        return self._train_dataset

    @property
    def eval_dataset(self) -> ListDataset:
        return self._eval_dataset

    @property
    def test_interactions(self) -> list[list[int]]:
        return self._test_interactions

    @property
    def train_interactions(self) -> list[list[int]]:
        return self._train_interactions

    @property
    def leave_k_out(self) -> int:
        return self._leave_k_out


class TrainModel:
    def __init__(self, vocab: Vocab, datasets: Datasets):
        self._vocab = vocab
        self._datasets = datasets
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metrics: dict[str, str] = {}

        self._create_model()
        self._create_trainer()

    def _create_model(self):
        # -hs 128 -ss 16 -is 64 -hl 8
        config = MambaConfig(
            hidden_size=32,
            num_hidden_layers=8,
            vocab_size=self._vocab.vocab_size,
            state_size=8,
            intermediate_size=32,
            use_mambapy=True,
            use_cache=False,
            pad_token_id=self._vocab.pad_id,
            bos_token_id=self._vocab.pad_id,  ## CLS
            eos_token_id=self._vocab.pad_id,  ## SEP
            expand=1,
        )
        self._model = MambaForCausalLM(config).to(self._device)

        assert self._model.num_parameters() > 1000

    def _create_trainer(self):
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            prediction_loss_only=True,
            save_strategy="best",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            learning_rate=2e-5,
            per_device_train_batch_size=256,
            per_device_eval_batch_size=64,
            num_train_epochs=5,
            weight_decay=0.01,
            use_cpu=False,
            data_seed=42,
            seed=42,
            disable_tqdm=False,
            full_determinism=True,
            save_total_limit=11,
            save_safetensors=False,
        )
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            data_collator=DataCollatorForCLMRec(self._vocab.pad_id),
            train_dataset=self._datasets.train_dataset,
            eval_dataset=self._datasets.eval_dataset,
        )

        self._trainer.train()

    def generate(
        self, max_new_tokens=None, dataset=None, batch_size: int = 512
    ) -> tuple[ListDataset, int, float]:
        if dataset is None:
            dataset = ListDataset(self._datasets.train_interactions)
        if max_new_tokens is None:
            max_new_tokens = self._datasets.leave_k_out

        self._at_k = max_new_tokens

        self._gconf = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=True,
            pad_token_id=self._vocab.pad_id,
            early_stopping="never",
            bad_words_ids=[
                [
                    self._vocab.pad_id,
                    self._vocab.unk_id,
                ]
            ],
        )

        inference = []
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=DataCollatorForCLMRec(self._vocab.pad_id),
            shuffle=False,
        )
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_inference = (
                    self._model.generate(
                        batch["input_ids"].to(self._device),
                        attention_mask=batch["attention_mask"].to(self._device),
                        generation_config=self._gconf,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )

                for lst in batch_inference:
                    lst2 = []
                    for el in lst:
                        if el != self._vocab.pad_id and el != self._vocab.unk_id:
                            lst2.append(el)
                    inference.append(lst2)

        self._inference_dataset = ListDataset(inference)
        self._metrics["distinct_inference_size"] = (
            self._inference_dataset.distinct_size()
        )
        self._metrics["cover_ratio"] = (
            1.0 * self._metrics["distinct_inference_size"] / dataset.distinct_size()
        )

        return (
            self._inference_dataset,
            self._metrics["distinct_inference_size"],
            self._metrics["cover_ratio"],
        )

    def pad(
        self, data: list[list[int]], pad_value: int, right_pad_margin: int
    ) -> list[list[int]]:
        """pad or truncate list with pad_value to right_pad_margin"""

        for lst in data:
            if len(lst) <= right_pad_margin:
                yield lst + [pad_value] * (right_pad_margin - len(lst))
            else:
                yield lst[:right_pad_margin]

    def ndcg(self, at_k=None) -> float:
        if at_k is None or at_k > self._at_k:
            at_k = self._at_k

        score = 1 * (
            np.array(
                list(
                    self.pad(
                        self._datasets.test_interactions, self._vocab.pad_id, at_k
                    )
                ),
                dtype=int,
            )
            == np.array(
                list(self.pad(self._inference_dataset.data, self._vocab.pad_id, at_k)),
                dtype=int,
            )
        )
        y_score = score.copy()
        score.sort(axis=1)
        y_true = np.fliplr(score)
        self._metrics[f"ndcg@{at_k}"] = ndcg_score(y_true, y_score, k=at_k).item()
        return self._metrics[f"ndcg@{at_k}"]

    def save(self, path: str = "./saved"):
        self._trainer.save_model(path)
        self._gconf.save_pretrained(path)

        with open(path + "/inference.obj", "wb") as fn:
            dill.dump(self._inference_dataset.data, fn)

        with open(path + "/vocab.obj", "wb") as fn:
            dill.dump(self._vocab, fn)

        with open(path + "/datasets.obj", "wb") as fn:
            dill.dump(self._datasets, fn)

        with open(path + "/metrics.json", "w") as fn:
            fn.write(json.dumps(self._metrics))
