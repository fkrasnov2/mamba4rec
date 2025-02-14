from collections.abc import Iterable
from collections import defaultdict


class Dataset:
    def __init__(
        self, user_item_interactions_it: Iterable[tuple[str, str]], leave_k_out: int = 5
    ):
        self._user_item_interactions_it = user_item_interactions_it
        self._leave_k_out = leave_k_out
        self._id2user = []
        self._item2id = {}
        self._id2item = {}
        self._train_item_ids, self._val_item_ids = self._make_leave_k_out()

    @property
    def item2id(self) -> dict:
        return self._item2id

    @property
    def id2item(self) -> dict:
        return self._id2item

    @property
    def id2user(self) -> list:
        return self._id2user

    @property
    def val_item_ids(self) -> list:
        return self._val_item_ids

    @property
    def train_item_ids(self) -> list:
        return self._train_item_ids

    @property
    def leave_k_out(self) -> int:
        return self._leave_k_out

    def _make_leave_k_out(self) -> tuple[list, list]:
        """
        Take user to item interactions and construct two lists: train and val
        """
        user_interactions = defaultdict(list)

        for user_item in self._user_item_interactions_it:
            user_interactions[user_item[0]].append(user_item[1])

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
        self._id2item = list(_items)

        train_item_ids = []
        val_item_ids = []

        for train, val in zip(train_items, val_items):
            train_item_ids.append(list(map(lambda item: self._item2id[item], train)))
            val_item_ids.append(list(map(lambda item: self._item2id[item], val)))

        print(f"{len(train_item_ids)=} {len(val_item_ids)=}")
        return train_item_ids, val_item_ids
