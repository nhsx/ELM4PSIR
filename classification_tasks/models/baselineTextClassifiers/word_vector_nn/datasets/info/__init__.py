from typing import Dict, Tuple

from . import ag_news


def get_label_map(dataset: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    if dataset == "ag_news":
        classes = ag_news.classes
    else:
        raise Exception("Dataset not supported: ", dataset)

    label_map = {k: v for v, k in enumerate(classes)}
    rev_label_map = {v: k for k, v in label_map.items()}

    return label_map, rev_label_map
