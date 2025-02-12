from collections import defaultdict
import  pytest

class DatasetML1M:
    def __init__(self, path_to_dataset: str, leave_k_out: int = 5):
        self.path_to_dataset = path_to_dataset
        self._leave_k_out = leave_k_out
        # self.load_users()
        # self.load_items()
        self._interactions  = []
        self._load_interactions()
        self._id2user = []
        self._item2id = {}
        self._id2item = {}
        self._make_leave_k_out()

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

    def _load_interactions(self):
        path_to_interactions = self.path_to_dataset + "/" + "ratings.dat"
        with open(path_to_interactions, "r") as fn:
            self._interactions = list(
                map(
                    lambda item_str: tuple(map(int, item_str.split("::")[:2])),
                    fn.read().strip("\n").split("\n"),
                )
            )

    def _make_leave_k_out(self) -> tuple[list, list]:
        user_interactions = defaultdict(list)

        for user_id, item_id in self._interactions:
            user_interactions[user_id].append(
                item_id
            )  ## strong: do not check user and items vocabs from dataset

        _items = set()
        train_items: list[list[int]] = []
        val_items: list[list[int]] = []

        for user_id, items in user_interactions.items():
            if len(items) < self._leave_k_out * 2:
                continue

            self._id2user.append(user_id)
            val_items.append(items[-self._leave_k_out :])
            train_items.append(items[: -self._leave_k_out])
            for it in items:
                _items.add(it)

        assert len(self._id2user) == len(set(self._id2user))

        self._item2id = {item: idx for idx, item in enumerate(_items)}
        train_item_ids = []
        val_item_ids = []

        for train, val in zip(train_items, val_items):
            train_item_ids.append(list(map(lambda item: self._item2id[item], train)))
            val_item_ids.append(list(map(lambda item: self._item2id[item], val)))

        print(f"{len(train_item_ids)=} {len(val_item_ids)=}")
        return train_item_ids, val_item_ids

    @property
    def item2id(self) -> dict:
        return self._item2id

    @property
    def id2item(self) -> dict:
        return self._id2item

    @property
    def id2user(self) -> list:
        return self._id2user

@pytest.fixture
def dataset_ml_1m():
    return DatasetML1M("dataset/ml-1m", leave_k_out=5)


def test_item2id(dataset_ml_1m):
    assert len(dataset_ml_1m.item2id) > 1000

def test_user2id(dataset_ml_1m):
    assert len(dataset_ml_1m.id2user) > 1000

