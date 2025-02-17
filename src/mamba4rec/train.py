import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from transformers import MambaConfig, MambaForCausalLM, Trainer, TrainingArguments


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __getitems__(self, idx_list):
        return [self.data[_] for _ in idx_list]


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
    def __init__(self, items: set):
        self._item2id = {item: idx for idx, item in enumerate(items)}
        self._item2id[self.pad_str] = len(self._item2id)
        self._item2id[self.unk_str] = len(self._item2id)

    def __getitem__(self, idx):
        return self.list(self._item2id.keys())[idx]

    @property
    def vocab_size(self) -> int:
        return len(self._item2id)

    @property
    def id2item(self) -> dict:
        return list(self._item2id.keys())

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
    def __init__(self, train_interactions: list, test_interactions: list):

        X_train, X_test, self._val_train, self._val_test = train_test_split(
            train_interactions,
            test_interactions,
            test_size=0.05,
            random_state=42,
        )

        self._train_dataset = ListDataset(X_train)
        self._eval_dataset = ListDataset(X_test)

    @property
    def train_dataset(self) -> ListDataset:
        return self._train_dataset

    @property
    def eval_dataset(self) -> ListDataset:
        return self._eval_dataset


class TrainModel:
    def __init__(self, vocab: Vocab, dl: Datasets):

        assert len(dl.train_dataset) > len(dl.eval_dataset)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # -hs 128 -ss 16 -is 64 -hl 8
        config = MambaConfig(
            hidden_size=32,
            num_hidden_layers=8,
            vocab_size=vocab.vocab_size,
            state_size=8,
            intermediate_size=32,
            use_mambapy=True,
            use_cache=False,
            pad_token_id=vocab.pad_id,
            bos_token_id=vocab.pad_id,  ## CLS
            eos_token_id=vocab.pad_id,  ## SEP
            expand=1,
        )
        model = MambaForCausalLM(config).to(device)

        assert model.num_parameters() > 1000

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="steps",
            prediction_loss_only=True,
            save_strategy="best",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            learning_rate=2e-5,
            per_device_train_batch_size=1024,
            per_device_eval_batch_size=1024,
            num_train_epochs=10,
            weight_decay=0.01,
            use_cpu=False,
            data_seed=42,
            seed=42,
            disable_tqdm=False,
            full_determinism=True,
            save_total_limit=5,
            save_safetensors=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForCLMRec(vocab.pad_id),
            train_dataset=dl.train_dataset,
            eval_dataset=dl.eval_dataset,
        )

        trainer.train()
        trainer.save_model("./saved")
