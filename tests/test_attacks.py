import copy
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import Dataset

from attacks import (AttackManager, LabelFlippedDataset, MaliciousSelector, build_label_map,
                     is_attack_round, parse_attack_config, poison_delta, poison_state_dict,
                     result_filename)

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "attack"


def load_cfg(name, **overrides):
    cfg = yaml.safe_load((CFG_DIR / f"{name}.yaml").read_text())
    for key, value in overrides.items():
        cur = cfg
        *path, last = key.split(".")
        for p in path:
            cur = cur[p]
        cur[last] = value
    return cfg


class Inner:
    classes = list(range(10))


class ToyDataset(Dataset):
    """FedACG의 DatasetSplitSubset처럼 class_dict와 dataset 속성을 가진 가짜 데이터셋."""

    def __init__(self, labels, tensor_labels=False):
        self.labels = labels
        self.tensor_labels = tensor_labels
        self.dataset = Inner()
        self.class_dict = {}
        for y in labels:
            self.class_dict[str(y)] = self.class_dict.get(str(y), 0) + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        y = self.labels[i]
        return torch.full((3,), float(i)), (torch.tensor(y) if self.tensor_labels else y)


# ---------------- 설정 ----------------
@pytest.mark.parametrize("name", ["none", "lf", "mp"])
def test_yaml_configs_parse(name):
    cfg = parse_attack_config(load_cfg(name))
    assert cfg.type == name


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        parse_attack_config(load_cfg("lf", type="backdoor"))
    with pytest.raises(ValueError):
        parse_attack_config(load_cfg("lf", start_round=10, end_round=5))
    with pytest.raises(KeyError):
        bad = load_cfg("lf")
        del bad["ratio"]
        parse_attack_config(bad)


# ---------------- 구간 공격 ----------------
def test_attack_window_half_open():
    assert not is_attack_round(99, 100, 200)
    assert is_attack_round(100, 100, 200)
    assert is_attack_round(199, 100, 200)
    assert not is_attack_round(200, 100, 200)
    assert is_attack_round(10_000, 100, None)


# ---------------- 악성 지정 ----------------
@pytest.mark.parametrize("ratio,expected", [(0.1, 10), (0.2, 20), (0.3, 30)])
def test_selector_count(ratio, expected):
    sel = MaliciousSelector(100, ratio, "fixed", seed=1)
    assert len(sel.malicious_set(0)) == expected


def test_selector_fixed_is_stable_and_seeded():
    a = MaliciousSelector(100, 0.2, "fixed", seed=1)
    b = MaliciousSelector(100, 0.2, "fixed", seed=1)
    c = MaliciousSelector(100, 0.2, "fixed", seed=2)
    assert a.malicious_set(0) == a.malicious_set(500) == b.malicious_set(3)
    assert a.malicious_set(0) != c.malicious_set(0)


def test_selector_random_changes_per_round_but_reproducible():
    a = MaliciousSelector(100, 0.2, "random", seed=1)
    b = MaliciousSelector(100, 0.2, "random", seed=1)
    sets = [a.malicious_set(r) for r in range(5)]
    assert len(set(sets)) > 1
    assert sets == [b.malicious_set(r) for r in range(5)]
    assert all(len(s) == 20 for s in sets)


def test_selector_does_not_touch_global_numpy_rng():
    np.random.seed(0)
    expected = np.random.choice(100, 10, replace=False)
    np.random.seed(0)
    sel = MaliciousSelector(100, 0.3, "random", seed=7)
    sel.malicious_set(3)
    got = np.random.choice(100, 10, replace=False)
    assert (expected == got).all()


# ---------------- 라벨 플리핑 ----------------
def test_label_maps():
    assert build_label_map(10, "reverse") == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    m = build_label_map(10, "pairs", pairs=[(3, 5), (5, 3)])
    assert m[3] == 5 and m[5] == 3 and m[0] == 0
    assert build_label_map(10, "all_to_one", target=2) == [2] * 10
    with pytest.raises(ValueError):
        build_label_map(10, "pairs", pairs=[(3, 11)])


@pytest.mark.parametrize("tensor_labels", [False, True])
def test_label_flipped_dataset(tensor_labels):
    base = ToyDataset([0, 1, 1, 9], tensor_labels=tensor_labels)
    ds = LabelFlippedDataset(base, build_label_map(10, "reverse"))
    assert len(ds) == 4
    labels = [int(ds[i][1]) for i in range(4)]
    assert labels == [9, 8, 8, 0]
    assert torch.equal(ds[2][0], base[2][0])              # 이미지는 그대로
    assert ds.class_dict == {"9": 1, "8": 2, "0": 1}      # 바뀐 라벨 기준으로 다시 셈
    assert ds.dataset.classes == list(range(10))          # 원래 속성 위임
    assert pickle.loads(pickle.dumps(ds))[0][1] == ds[0][1]  # spawn 모드용 pickle


# ---------------- 모델 오염 ----------------
def _states():
    torch.manual_seed(0)
    g = {"w": torch.randn(4, 3), "b": torch.randn(3), "bn.num_batches_tracked": torch.tensor(5)}
    l = {k: (v + 0.1 * torch.randn_like(v) if v.is_floating_point() else v + 1) for k, v in g.items()}
    return l, g


def test_poison_delta_scale_and_sign():
    delta = {"w": torch.ones(2, 2), "n": torch.tensor(3)}
    assert torch.allclose(poison_delta(delta, 3.0, False)["w"], 3 * torch.ones(2, 2))
    assert torch.allclose(poison_delta(delta, 3.0, True)["w"], -3 * torch.ones(2, 2))
    assert poison_delta(delta, 3.0, True)["n"].item() == 3   # 정수 버퍼는 그대로


def test_poison_state_dict_consistent():
    l, g = _states()
    new_state, new_delta = poison_state_dict(l, g, scale=3.0, sign_flip=True)
    for k in ("w", "b"):
        assert torch.allclose(new_delta[k], -3.0 * (l[k] - g[k]), atol=1e-6)
        assert torch.allclose(new_state[k] - g[k], new_delta[k], atol=1e-6)
    assert new_state["bn.num_batches_tracked"].item() == l["bn.num_batches_tracked"].item()


def test_poison_noise_reproducible():
    l, g = _states()
    gen = lambda: torch.Generator().manual_seed(3)
    a, _ = poison_state_dict(l, g, 1.0, False, noise_std=0.5, generator=gen())
    b, _ = poison_state_dict(l, g, 1.0, False, noise_std=0.5, generator=gen())
    assert torch.allclose(a["w"], b["w"]) and not torch.allclose(a["w"], l["w"])


# ---------------- 매니저 ----------------
def make_manager(name, **overrides):
    return AttackManager(parse_attack_config(load_cfg(name, **overrides)),
                         num_clients=100, num_classes=10, seed=1)


def test_manager_none_is_noop():
    m = make_manager("none")
    l, g = _states()
    info = m.begin_round(150, range(100))
    assert not info.attack_active and info.n_malicious == 0
    base = ToyDataset([1, 2])
    assert m.maybe_flip_labels(0, 150, base) is base
    assert m.maybe_poison(0, 150, l, g) is l


def test_manager_lf_only_malicious_in_window():
    m = make_manager("lf")
    mal = sorted(m.selector.malicious_set(0))
    benign = next(c for c in range(100) if c not in mal)
    base = ToyDataset([1, 2])
    assert isinstance(m.maybe_flip_labels(mal[0], 150, base), LabelFlippedDataset)
    assert m.maybe_flip_labels(mal[0], 50, base) is base      # 구간 전: 정상 행동
    assert m.maybe_flip_labels(mal[0], 200, base) is base     # 구간 후: 정상 행동
    assert m.maybe_flip_labels(benign, 150, base) is base
    l, g = _states()
    assert m.maybe_poison(mal[0], 150, l, g) is l             # lf는 Δ를 건드리지 않음


def test_manager_mp_only_malicious_in_window():
    m = make_manager("mp")
    mal = sorted(m.selector.malicious_set(0))
    l, g = _states()
    poisoned = m.maybe_poison(mal[0], 150, l, g)
    assert torch.allclose(poisoned["w"] - g["w"], -3.0 * (l["w"] - g["w"]), atol=1e-6)
    assert m.maybe_poison(mal[0], 250, l, g) is l
    base = ToyDataset([1, 2])
    assert m.maybe_flip_labels(mal[0], 150, base) is base


def test_round_info_counts_f_r():
    m = make_manager("lf", ratio=0.3)
    selected = list(range(10))
    info = m.begin_round(150, selected)
    expected = len(set(selected) & m.selector.malicious_set(150))
    assert info.n_selected == 10 and info.n_malicious == expected
    row = info.as_log_row()
    assert set(row) == {"round", "attack_active", "n_selected", "n_malicious", "malicious_ids"}


def test_manager_from_omegaconf_args():
    omegaconf = pytest.importorskip("omegaconf")
    args = omegaconf.OmegaConf.create({"seed": 1, "trainer": {"num_clients": 100},
                                       "attack": load_cfg("mp", ratio=0.1)})
    m = AttackManager.from_args(args, num_classes=10)
    assert m.cfg.type == "mp" and len(m.selector.malicious_set(0)) == 10
    args_no_attack = omegaconf.OmegaConf.create({"seed": 1, "trainer": {"num_clients": 100}})
    assert not AttackManager.from_args(args_no_attack, num_classes=10).enabled


def test_manager_survives_pickle():
    m = make_manager("lf", selection="random")
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.selector.malicious_set(7) == m.selector.malicious_set(7)


# ---------------- 파일 이름 ----------------
def test_result_filename():
    assert result_filename("FedACG", "Krum", "lf", 20, 1) == "fedacg_krum_lf_20_s1.csv"
    assert result_filename("fedavg", "multi-krum", "mp", 10, 3) == "fedavg_multikrum_mp_10_s3.csv"
    assert result_filename("fedacg", "trimmed_mean", "none", 0, 2) == "fedacg_trimmedmean_none_0_s2.csv"
