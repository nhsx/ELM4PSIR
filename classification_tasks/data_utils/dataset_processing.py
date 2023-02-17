import os
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def count_trainable_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def encode_classes(df, meta_df, label_col=None):
    """
    Function to encode non-integer categories or classes into integers or floats to be
    used by downstream models
    """

    # get classes as unique/set of the label col - we sort to make int labels stay in
    # order if possible

    classes = sorted(df[label_col].unique())

    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}

    # and also get a list of the actual labels in order to return
    class_list = list(class_to_idx.keys())

    return class_list, idx_to_class, class_to_idx


def convert_to_binary_classes(df, split_value, label_col="label"):
    """
    Function to take a dataframe with a class label with range 0-N where N > 1 i.e.
    more than 2 classes and convert to a binary 0/1 class problem given a split value
    in range 0-N.
    """

    assert split_value <= df[label_col].max() and split_value >= df[label_col].min(), (
        f"it appears that the split value given:{split_value} is outside of the range "
        "of the dfs label min:{df[label_col].min()} and max:{df[label_col].max()}"
    )

    # make the 0 class label
    df[label_col].values[df[label_col] < split_value] = 0
    # make the 1 class label
    df[label_col].values[df[label_col] >= split_value] = 1

    print(f"Class value counts: {df[label_col].value_counts()}")
    return df


def balance_dataset(self, df, random_state=42):
    """
    Function to balance the training dataset - won't bother with the valid and test
    sets
    """

    # slightly clunky but works
    g = df.groupby("hospital_expire_flag")
    g = pd.DataFrame(
        g.apply(
            lambda x: x.sample(g.size().min(), random_state=random_state).reset_index(
                drop=True
            )
        )
    )
    g.reset_index(drop=True, inplace=True)

    return g.sample(frac=1, random_state=random_state)


class FewShotSampler(object):
    """
    Few-shot learning is an important scenario this is sampler that samples few
    examples over each class.
    Args:
        num_examples_total(:obj:`int`, optional): Sampling strategy ``I``: Use total
        number of examples for few-shot sampling.

        num_examples_per_label(:obj:`int`, optional): Sampling strategy ``II``: Use the
        number of examples for each label for few-shot sampling.

        also_sample_dev(:obj:`bool`, optional): Whether to apply the sampler to the dev
        data.

        num_examples_total_dev(:obj:`int`, optional): Sampling strategy ``I``: Use
        total number of examples for few-shot sampling.

        num_examples_per_label_dev(:obj:`int`, optional): Sampling strategy ``II``:
        Use the number of examples for each label for few-shot sampling.
    """

    def __init__(
        self,
        num_examples_total: Optional[int] = None,
        num_examples_per_label: Optional[int] = None,
        also_sample_dev: Optional[bool] = False,
        num_examples_total_dev: Optional[int] = None,
        num_examples_per_label_dev: Optional[int] = None,
        label_col="label",
    ):
        if num_examples_total is None and num_examples_per_label is None:
            raise ValueError(
                "num_examples_total and num_examples_per_label can't be both None."
            )
        elif num_examples_total is not None and num_examples_per_label is not None:
            raise ValueError(
                "num_examples_total and num_examples_per_label can't be both set."
            )

        if also_sample_dev:
            if (
                num_examples_total_dev is not None
                and num_examples_per_label_dev is not None
            ):
                raise ValueError(
                    "num_examples_total and num_examples_per_label can't be both set."
                )
            elif num_examples_total_dev is None and num_examples_per_label_dev is None:
                logger.warning(
                    (
                        "specify neither num_examples_total_dev nor "
                        "num_examples_per_label_dev, set to default (equal to train "
                        "set setting)."
                    )
                )
                self.num_examples_total_dev = num_examples_total
                self.num_examples_per_label_dev = num_examples_per_label
            else:
                self.num_examples_total_dev = num_examples_total_dev
                self.num_examples_per_label_dev = num_examples_per_label_dev

        self.num_examples_total = num_examples_total
        self.num_examples_per_label = num_examples_per_label
        self.also_sample_dev = also_sample_dev
        self.label_col = label_col

    def __call__(
        self,
        dataset: Union[Dataset, List],
        valid_dataset: Optional[Union[Dataset, List]] = None,
        seed: Optional[int] = None,
    ) -> Union[Dataset, List]:
        """
        The ``__call__`` function of the few-shot sampler.
        Args:
            dataset (:obj:`Dictionary or dataframe`): The train dataset for the sampler.

            valid_dataset (:obj:`Union[Dataset, List]`, optional): The valid datset for
            the sampler. Default to None.

            seed (:obj:`int`, optional): The random seed for the sampling.

        Returns:
            :obj:`(Union[Dataset, List], Union[Dataset, List])`: The sampled dataset
            (dataset, valid_dataset), whose type is identical to the input.
        """
        if valid_dataset is None:
            if self.also_sample_dev:
                return self._sample(dataset, seed, sample_twice=True)
            else:
                dataset = self._sample(dataset, seed, sample_twice=False)
                return pd.DataFrame(dataset)
        else:
            dataset = self._sample(dataset, seed)
            if self.also_sample_dev:
                valid_dataset = self._sample(valid_dataset, seed)
            return pd.DataFrame(dataset)

    def _sample(
        self,
        data: Union[Dataset, List],
        seed: Optional[int],
        sample_twice=False,
    ) -> Union[Dataset, List]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        indices = [i for i in range(len(data))]

        if self.num_examples_per_label is not None:
            assert (
                self.label_col in data[0].keys()
            ), "sample by label requires the data has the label_col attribute."
            labels = [x[self.label_col] for x in data]
            selected_ids = self.sample_per_label(
                indices, labels, self.num_examples_per_label
            )  # TODO fix: use num_examples_per_label_dev for dev
        else:
            selected_ids = self.sample_total(indices, self.num_examples_total)

        if sample_twice:
            selected_set = set(selected_ids)
            remain_ids = [i for i in range(len(data)) if i not in selected_set]
            if self.num_examples_per_label_dev is not None:
                assert (
                    self.label_col in data[0].keys()
                ), "sample by label requires the data has a 'label' attribute."
                remain_labels = [
                    x[self.label_col]
                    for idx, x in enumerate(data)
                    if idx not in selected_set
                ]
                selected_ids_dev = self.sample_per_label(
                    remain_ids, remain_labels, self.num_examples_per_label_dev
                )
            else:
                selected_ids_dev = self.sample_total(
                    remain_ids, self.num_examples_total_dev
                )

            if isinstance(data, Dataset):
                return Subset(data, selected_ids), Subset(data, selected_ids_dev)
            elif isinstance(data, List):
                return [data[i] for i in selected_ids], [
                    data[i] for i in selected_ids_dev
                ]

        else:
            if isinstance(data, Dataset):
                return Subset(data, selected_ids)
            elif isinstance(data, List):
                return [data[i] for i in selected_ids]

    def sample_total(self, indices: List, num_examples_total):
        """
        Use the total number of examples for few-shot sampling (Strategy ``I``).

        Args:
            indices(:obj:`List`): The random indices of the whole datasets.
            num_examples_total(:obj:`int`): The total number of examples.

        Returns:
            :obj:`List`: The selected indices with the size of ``num_examples_total``.

        """
        self.rng.shuffle(indices)
        selected_ids = indices[:num_examples_total]
        # logger.info("Selected examples (mixed) {}".format(selected_ids))
        return selected_ids

    def sample_per_label(self, indices: List, labels, num_examples_per_label):
        """
        Use the number of examples per class for few-shot sampling (Strategy ``II``).
        If the number of examples is not enough, a warning will pop up.

        Args:
            indices(:obj:`List`): The random indices of the whole datasets.
            labels(:obj:`List`): The list of the labels.
            num_examples_per_label(:obj:`int`): The total number of examples for each
            class.

        Returns:
            :obj:`List`: The selected indices with the size of ``num_examples_total``.
        """

        ids_per_label = defaultdict(list)
        selected_ids = []
        for idx, label in zip(indices, labels):
            ids_per_label[label].append(idx)
        for label, ids in ids_per_label.items():
            tmp = np.array(ids)
            self.rng.shuffle(tmp)
            if len(tmp) < num_examples_per_label:
                logger.info(
                    "Not enough examples of label {} can be sampled".format(label)
                )
            selected_ids.extend(tmp[:num_examples_per_label].tolist())
        selected_ids = np.array(selected_ids)
        self.rng.shuffle(selected_ids)
        selected_ids = selected_ids.tolist()
        # logger.info("Selected examples {}".format(selected_ids))
        return selected_ids
