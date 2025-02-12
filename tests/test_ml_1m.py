import sys
sys.path.append("src")

from mamba4rec import Dataset
import  pytest
from collections.abc import Iterable 

class DatasetML1M (Dataset):
    def __init__(self, path_to_dataset: str, leave_k_out: int = 5):
        self._path_to_dataset = path_to_dataset
        self._leave_k_out = leave_k_out
        # self.load_users()
        # self.load_items()
        super().__init__(self._load_interactions(path_to_dataset), leave_k_out)

    #    def load_users(self):
    #        path_to_user_vocab = self.path_to_dataset + "/" + "users.dat"
    #        with open(path_to_user_vocab, "r") as fn:
    #            self.users_vocab = list(
    #                map(
    #                    lambda user_str: user_str.split("::").pop(0),
    #                    fn.read().strip("\n").split("\n"),
    #                )
    #            )
    #
    #    def load_items(self):
    #        path_to_item_vocab = self.path_to_dataset + "/" + "movies.dat"
    #        with open(path_to_items_vocab, "r") as fn:
    #            self.items_vocab = list(
    #                map(
    #                    lambda item_str: item_str.split("::").pop(0),
    #                    fn.read().strip("\n").split("\n"),
    #                )
    #            )

    def _load_interactions(self, path_to_dataset) -> Iterable[tuple[int,int]]:
        path_to_interactions = path_to_dataset + "/" + "ratings.dat"
        with open(path_to_interactions, "r") as file:
            for line in file:
                yield tuple(map(int, line.strip("\n").split("::")[:2]))
           
@pytest.fixture
def dataset_ml_1m():
    return DatasetML1M("dataset/ml-1m", leave_k_out=5)

def test_item2id(dataset_ml_1m):
    assert len(dataset_ml_1m.item2id) > 1000

def test_user2id(dataset_ml_1m):
    assert len(dataset_ml_1m.id2user) > 1000

def test_id2item(dataset_ml_1m):
    assert len(dataset_ml_1m.id2item) > 1000

def test_val_item_ids(dataset_ml_1m):
    assert len(dataset_ml_1m.val_item_ids) == len(dataset_ml_1m.id2user)

def test_train_item_ids(dataset_ml_1m):
    assert len(dataset_ml_1m.train_item_ids) == len(dataset_ml_1m.id2user) 

def test_train_item_ids(dataset_ml_1m):
    assert len(dataset_ml_1m.train_item_ids[512]) >= dataset_ml_1m.leave_k_out

def test_val_item_ids(dataset_ml_1m):
    assert len(dataset_ml_1m.val_item_ids[512]) == dataset_ml_1m.leave_k_out




