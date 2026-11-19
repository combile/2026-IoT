"""라벨 플리핑: 악성 클라이언트가 바뀐 라벨로 학습한다."""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def build_label_map(num_classes: int, mode: str,
                    pairs: Sequence[Tuple[int, int]] = (),
                    target: Optional[int] = None) -> List[int]:
    """원래 라벨 y → 바뀐 라벨 map[y].

    - reverse   : y → C-1-y (모든 라벨을 바꿈)
    - pairs     : 지정한 (src, dst)만 바꿈. 맞바꾸려면 두 방향을 모두 적는다
    - all_to_one: 모든 라벨 → target
    """
    if num_classes < 2:
        raise ValueError("num_classes는 2 이상이어야 한다")
    if mode == "reverse":
        return [num_classes - 1 - y for y in range(num_classes)]
    if mode == "pairs":
        if not pairs:
            raise ValueError("pairs 모드는 label_flip.pairs가 필요하다")
        mapping = list(range(num_classes))
        for src, dst in pairs:
            for v in (src, dst):
                if not 0 <= v < num_classes:
                    raise ValueError(f"라벨 범위를 벗어났다: {v}")
            mapping[src] = dst
        return mapping
    if mode == "all_to_one":
        if target is None or not 0 <= target < num_classes:
            raise ValueError(f"all_to_one 모드의 target이 잘못됐다: {target}")
        return [target] * num_classes
    raise ValueError(f"알 수 없는 label_flip 모드: {mode}")


class LabelFlippedDataset(Dataset):
    """로컬 데이터셋을 감싸 라벨만 바꾼다. 이미지와 순서는 그대로다.

    FedACG의 Client.setup이 쓰는 `class_dict`는 바뀐 라벨 기준으로 다시 센다.
    그 밖의 속성(`dataset`, `indices` 등)은 원래 데이터셋으로 넘긴다.
    """

    def __init__(self, base: Dataset, label_map: Sequence[int]):
        self.base = base
        self.label_map = list(label_map)
        self.class_dict = self._remap_class_dict(getattr(base, "class_dict", None))

    def _remap_class_dict(self, class_dict: Optional[Dict[str, int]]) -> Dict[str, int]:
        if class_dict is None:
            return {}
        out: Dict[str, int] = {}
        for label, count in class_dict.items():
            new = str(self.label_map[int(label)])
            out[new] = out.get(new, 0) + count
        return out

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        if torch.is_tensor(label):
            return image, torch.as_tensor(self.label_map[int(label.item())], dtype=label.dtype)
        return image, self.label_map[int(label)]

    def __getattr__(self, name):
        # pickle 복원 중 base가 아직 없을 때 무한 재귀를 막는다
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)
