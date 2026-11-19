import copy
import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from defenses import (aggregate_deltas, assumed_f, coordinate_median, krum_condition_ok,
                      normalize_name, robust_aggregate, trimmed_mean)

KEYS = {"conv.weight": (4, 3), "fc.bias": (5,)}


def make_args(aggregator, momentum=0.85, fedacg=True, ratio=0.3, defense=True):
    cfg = {"server": {"aggregator": aggregator, "momentum": momentum, "FedACG": fedacg}}
    if defense:
        cfg["defense"] = {"assumed_ratio": ratio}
    return OmegaConf.create(cfg)


class FakeServer:
    """FedACG servers/base.py의 ServerM과 같은 상태(global_delta, global_momentum)를 가진 가짜 서버."""

    def __init__(self, args, model_dict):
        self.args = args
        self.global_delta = {k: torch.zeros_like(v) for k, v in model_dict.items()}
        self.global_momentum = {k: torch.randn_like(v) for k, v in model_dict.items()}


def make_round(n, seed=0, attackers=0, scale=-10.0):
    """(model_dict, local_weights, local_deltas, 정상 방향). 뒤쪽 attackers명은 Δ를 scale배로 조작한다."""
    g = torch.Generator().manual_seed(seed)
    model = {k: torch.randn(s, generator=g) for k, s in KEYS.items()}
    honest_dir = {k: torch.randn(s, generator=g) for k, s in KEYS.items()}
    weights, deltas = {k: [] for k in KEYS}, {k: [] for k in KEYS}
    for i in range(n):
        for k, s in KEYS.items():
            d = honest_dir[k] + 0.1 * torch.randn(s, generator=g)
            if i >= n - attackers:
                d = scale * d
            deltas[k].append(d)
            weights[k].append(model[k] + d)
    return model, weights, deltas, honest_dir


def original_serverm(args, server, local_weights, local_deltas, C):
    """FedACG 원본 ServerM.aggregate (servers/base.py)를 그대로 옮긴 기준 구현."""
    local_weights = {k: sum(v) / C for k, v in local_weights.items()}
    if args.server.momentum > 0:
        if not args.server.get("FedACG"):
            for k in local_weights:
                local_weights[k] += args.server.momentum * server.global_momentum[k]
        for k in local_deltas:
            server.global_delta[k] = sum(local_deltas[k]) / C
            server.global_momentum[k] = args.server.momentum * server.global_momentum[k] + server.global_delta[k]
    return local_weights


def test_names():
    assert normalize_name("Multi-Krum") == "multikrum"
    assert normalize_name("trimmed_mean") == "trimmedmean"
    assert normalize_name(None) == "mean"
    with pytest.raises(ValueError):
        normalize_name("bulyan")


def test_assumed_f_and_krum_condition():
    assert assumed_f(0.3, 10) == 3 and krum_condition_ok(10, 3)
    assert assumed_f(0.2, 5) == 1 and krum_condition_ok(5, 1)
    assert assumed_f(0.3, 5) == 1
    assert not krum_condition_ok(5, 2)
    with pytest.raises(ValueError):
        assumed_f(0.5, 10)


def test_median_matches_numpy_for_even_and_odd():
    for n in (4, 5, 10):
        x = torch.randn(n, 3, 2)
        np.testing.assert_allclose(coordinate_median(x).numpy(), np.median(x.numpy(), axis=0), rtol=1e-6)


def test_trimmed_mean_drops_extremes():
    x = torch.tensor([[1.0], [2.0], [3.0], [100.0], [-100.0]])
    assert trimmed_mean(x, 1).item() == pytest.approx(2.0)
    assert trimmed_mean(x, 0).item() == pytest.approx(x.mean().item())
    assert trimmed_mean(x, 9).item() == pytest.approx(2.0)  # 너무 크면 중앙값까지만 자른다


@pytest.mark.parametrize("name", ["krum", "multikrum", "trimmedmean", "median"])
def test_robust_rules_resist_sign_flip_attack(name):
    n, attackers = 10, 3
    _, _, deltas, honest = make_round(n, attackers=attackers)
    D, used = aggregate_deltas(name, deltas, f=3)
    mean_D, _ = aggregate_deltas("mean", deltas, f=3)
    for k in KEYS:
        err = (D[k] - honest[k]).norm()
        assert err < 0.5 * (mean_D[k] - honest[k]).norm()
    if name in ("krum", "multikrum"):
        assert all(i < n - attackers for i in used)          # 공격자는 뽑히지 않는다
        assert len(used) == (1 if name == "krum" else n - 3)


@pytest.mark.parametrize("fedacg", [True, False])
def test_multikrum_f0_equals_original_fedacg_and_fedavgm(fedacg):
    """f = 0이면 Multi-Krum은 n명 전원 평균 = 원본 집계와 같아야 한다 (모델과 모멘텀 모두)."""
    n = 6
    model, weights, deltas, _ = make_round(n, seed=1)
    args = make_args("multikrum", fedacg=fedacg, ratio=0.0)
    ours = FakeServer(args, model)
    ref = copy.deepcopy(ours)
    new = robust_aggregate(ours, weights, deltas, list(range(n)), copy.deepcopy(model))
    expect = original_serverm(args, ref, copy.deepcopy(weights), deltas, n)
    for k in KEYS:
        torch.testing.assert_close(new[k], expect[k])
        torch.testing.assert_close(ours.global_momentum[k], ref.global_momentum[k])


def test_fedavg_without_momentum():
    n = 4
    model, weights, deltas, _ = make_round(n, seed=2)
    args = make_args("multikrum", momentum=-1, fedacg=False, ratio=0.0)
    server = type("S", (), {"args": args})()
    new = robust_aggregate(server, weights, deltas, list(range(n)), copy.deepcopy(model))
    for k in KEYS:
        torch.testing.assert_close(new[k], sum(weights[k]) / n)


def test_integer_buffer_is_averaged():
    deltas = {"w": [torch.ones(2), 3 * torch.ones(2), 5 * torch.ones(2)],
              "bn.num_batches_tracked": [torch.tensor(3), torch.tensor(3), torch.tensor(3)]}
    D, _ = aggregate_deltas("median", deltas, f=1)
    assert D["w"].tolist() == [3.0, 3.0]
    assert float(D["bn.num_batches_tracked"]) == 3.0


def test_selected_client_ids_are_recorded():
    n = 10
    model, weights, deltas, _ = make_round(n, attackers=3)
    server = FakeServer(make_args("krum"), model)
    ids = [40 + i for i in range(n)]
    robust_aggregate(server, weights, deltas, ids, copy.deepcopy(model))
    assert len(server.last_selected) == 1 and server.last_selected[0] < 47


def test_missing_defense_config_raises():
    model, weights, deltas, _ = make_round(4)
    server = FakeServer(make_args("median", defense=False), model)
    with pytest.raises(KeyError):
        robust_aggregate(server, weights, deltas, list(range(4)), copy.deepcopy(model))


def test_krum_condition_warning(caplog):
    n = 5
    model, weights, deltas, _ = make_round(n)
    server = FakeServer(make_args("krum", ratio=0.4), model)   # f = 2, 5 > 6 위반
    with caplog.at_level(logging.WARNING):
        robust_aggregate(server, weights, deltas, list(range(n)), copy.deepcopy(model))
    assert "n > 2f + 2" in caplog.text
