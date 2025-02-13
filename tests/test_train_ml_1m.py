
from .test_ml_1m import  DatasetML1M 
import pytest

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import MambaConfig, MambaForCausalLM
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import os
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True

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
        #labels = padded_ids[:, -1]#.unsqueeze(dim=1)
        #padded_ids = padded_ids[:,:-1]
        #print(f"{padded_ids.shape=} {labels.shape=}")
        return {
            "input_ids": padded_ids,
            "attention_mask": padded_ids != self.pad_id,
            "labels": padded_ids,
        }

    def __call__(self, batch_of_ids):
        return self.mask_ids_batch(batch_of_ids)

class CreateDataloaders:
    def __init__(self, dataset : DatasetML1M ):
        self._dataset = dataset
        self._pad_str = "[PAD]"
        self._pad_id = len(self._dataset.item2id)
        self._dataset.item2id[self._pad_str] = self._pad_id
        self._unk_id = len(self._dataset.item2id)
        self._unk_str = "[UNK]"
        self._dataset.item2id[self._unk_str] = self._unk_id
        self._vocab_size = len(self._dataset.item2id)


        X_train, X_test, self._val_train, self._val_test = train_test_split(self._dataset.train_item_ids, self._dataset.val_item_ids,  test_size=0.05, random_state=42)

        self._train_dataset = ListDataset(X_train)
        self._test_dataset = ListDataset(X_test)

    def save_model(self):
        #self._fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object = Tokenizer(WordLevel(self._dataset.item2id, self._unk_str)),   pad_token=self._pad_str, unk_token=self._unk_str,)
        #fast_tokenizer.save_pretrained(args.hf_path)
        #model.save_pretrained(args.hf_path + "/last")
        #fast_tokenizer.save_pretrained(args.hf_path + "/last")      
        pass

@pytest.fixture
def dataset_ml_1m():
    return DatasetML1M("dataset/ml-1m", leave_k_out=2)

def test_model_train(dataset_ml_1m):

    dl = CreateDataloaders(dataset_ml_1m)
    assert len(dl._train_dataset) > len (dl._test_dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = MambaConfig(
        hidden_size=8,
        num_hidden_layers=2,
        vocab_size=dl._vocab_size,
        state_size=4,
        intermediate_size=4,
        use_mambapy=True,
        use_cache=False,
        pad_token_id=dl._pad_id,
        bos_token_id=dl._pad_id,  ## CLS
        eos_token_id=dl._pad_id,  ## SEP
        expand=1,
    )
    #print(config, flush=True)
    model = MambaForCausalLM(config).to(device)
    
    assert model.num_parameters() > 1000

    #print(model.__class__, model.num_parameters(), "parameters!", flush=True)
    #print(model, flush=True)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        use_cpu = True,
        data_seed = 42,
        seed = 42,
        disable_tqdm = True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForCLMRec(dl._pad_id),
        train_dataset=dl._train_dataset,
        eval_dataset=dl._test_dataset,
    )
        
    assert trainer.train()
    assert trainer.evaluate()


