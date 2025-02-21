import sys

sys.path.append("src")

from mamba4rec import Vocab
import pytest


@pytest.fixture
def vocab():
    return Vocab({"nike air": (0, "Nice Aier"), "xiaomi": (1, "Xiaome")})


def test_item_id_to_raw_item(vocab):
    assert vocab.item_id_to_raw_item(0) == "Nice Aier"


def test_item2raw_item(vocab):
    assert vocab.item2raw_item("nike air") == "Nice Aier"


def test_vocab_size(vocab):
    assert vocab.vocab_size == 4


def test_id2item(vocab):
    assert vocab.id2item == {0: "nike air", 1: "xiaomi", 2: "[PAD]", 3: "[UNK]"}


def test_item2id(vocab):
    assert vocab.item2id == {"nike air": 0, "xiaomi": 1, "[PAD]": 2, "[UNK]": 3}


def test_pad_id(vocab):
    assert vocab.pad_id == 2


def test_unk_id(vocab):
    assert vocab.unk_id == 3
