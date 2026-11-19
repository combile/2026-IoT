"""이어 돌리기: 끊긴 실행을 마지막 체크포인트부터 같은 결과로 다시 시작한다.

체크포인트(<결과 이름>.pt)에 담는 것
- round          : 마지막으로 끝낸 라운드 r. 다시 시작하면 r+1부터 돈다
- model          : 그 라운드 집계 후 글로벌 모델
- server         : 서버 상태 텐서(global_momentum = m, global_delta, global_v 중 있는 것)
- ema            : EMA 정확도
- rng            : python random, numpy, torch(CPU, CUDA) 난수 상태
- config         : 실행 설정. 다시 시작할 때 설정이 바뀌었는지 확인한다

FedACG 원본 체크포인트는 모델만 저장해서 이어 돌리면 m이 0부터 다시 시작한다.
그러면 FedACG 결과가 끊기지 않은 실행과 달라지므로 서버 상태와 난수 상태까지 저장한다.
"""
import copy
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
SERVER_STATE_ATTRS = ("global_momentum", "global_delta", "global_v")

# 바뀌어도 결과에 영향이 없거나(경로, 화면 출력), 늘려서 이어 돌릴 수 있는 설정
IGNORED_CONFIG_KEYS = (
    "trainer.global_rounds", "resume", "metrics.console_level", "metrics.results_dir",
    "dataset.path", "dataset.client_path", "checkpoint_path", "log_dir", "wandb",
    "num_workers", "pin_memory", "save_freq", "main_gpu", "load_model_path", "wandb_resume_id",
)


# ---------------- 난수 상태 ----------------
def capture_rng() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------- 설정 비교 ----------------
def _to_plain(args: Any) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(args):
            return OmegaConf.to_container(args, resolve=True)
    except ImportError:
        pass
    return copy.deepcopy(dict(args))


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def _ignored(key: str) -> bool:
    return any(key == k or key.startswith(k + ".") for k in IGNORED_CONFIG_KEYS)


def config_diff(saved: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """결과에 영향을 주는 설정 중 달라진 키 목록."""
    a, b = _flatten(saved), _flatten(current)
    keys = sorted(k for k in set(a) | set(b) if not _ignored(k))
    return [f"{k}: {a.get(k, '<없음>')} -> {b.get(k, '<없음>')}" for k in keys if a.get(k) != b.get(k)]


# ---------------- 체크포인트 ----------------
class ResumeManager:
    def __init__(self, path, every: int, keep_final: bool, strict: bool, config: Dict[str, Any]):
        if every < 1:
            raise ValueError(f"resume.every는 1 이상이어야 한다: {every}")
        self.path = Path(path)
        self.every = int(every)
        self.keep_final = bool(keep_final)
        self.strict = bool(strict)
        self.config = config

    @classmethod
    def from_args(cls, args: Any, result_name: str) -> Optional["ResumeManager"]:
        """args.resume 설정으로 만든다. 설정이 없거나 enabled=false면 None(이어 돌리기 끔)."""
        cfg = args.get("resume") if hasattr(args, "get") else None
        if cfg is None or not cfg.get("enabled"):
            return None
        ckpt_dir = cfg.get("dir")
        if not ckpt_dir:
            ckpt_dir = Path(args.metrics.results_dir) / "checkpoints"
        path = Path(ckpt_dir) / Path(result_name).with_suffix(".pt")
        return cls(path, every=cfg.every, keep_final=cfg.keep_final, strict=cfg.strict,
                   config=_to_plain(args))

    # ---- 불러오기 ----
    def load(self) -> Optional[Dict[str, Any]]:
        """체크포인트가 있으면 내용을, 없으면 None을 돌려준다. 설정이 다르면 strict일 때 멈춘다."""
        if not self.path.exists():
            return None
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if state.get("version") != FORMAT_VERSION:
            raise RuntimeError(f"체크포인트 형식이 다르다: {self.path} (version={state.get('version')})")
        diff = config_diff(state["config"], self.config)
        if diff:
            msg = (f"체크포인트와 지금 설정이 다르다: {self.path}\n  " + "\n  ".join(diff) +
                   "\n같은 실험이 아니면 체크포인트를 지우거나 resume.enabled=false로 처음부터 돌린다.")
            if self.strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        return state

    @staticmethod
    def start_round(state: Optional[Dict[str, Any]]) -> int:
        return 0 if state is None else int(state["round"]) + 1

    @staticmethod
    def restore(state: Dict[str, Any], model: torch.nn.Module, server: Any) -> None:
        """모델과 서버 상태를 되돌린다. 난수 상태는 restore_rng로 학습 직전에 따로 되돌린다."""
        model.load_state_dict(state["model"])
        for attr, tensors in state["server"].items():
            target = getattr(server, attr, None)
            if target is None:
                raise RuntimeError(f"체크포인트에 server.{attr}가 있는데 지금 서버에는 없다")
            for k, v in tensors.items():
                target[k] = v.to(target[k].device) if k in target else v

    # ---- 저장 ----
    def should_save(self, rnd: int, last_round: int) -> bool:
        """every라운드마다 저장한다. 마지막 라운드는 keep_final일 때만 저장한다(아니면 곧 지운다)."""
        if rnd >= last_round:
            return self.keep_final
        return (rnd + 1) % self.every == 0

    def save(self, rnd: int, model: torch.nn.Module, server: Any, ema: Optional[float]) -> None:
        server_state = {}
        for attr in SERVER_STATE_ATTRS:
            tensors = getattr(server, attr, None)
            if isinstance(tensors, dict):
                server_state[attr] = {k: v.detach().cpu().clone() for k, v in tensors.items()}
        state = {
            "version": FORMAT_VERSION,
            "round": int(rnd),
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "server": server_state,
            "ema": ema,
            "rng": capture_rng(),
            "config": self.config,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.path)   # 저장 중에 끊겨도 이전 체크포인트는 온전하다
        logger.info(f"[Resume] saved round {rnd} -> {self.path}")

    def finish(self) -> None:
        """학습이 끝까지 끝났을 때 부른다. keep_final=false면 체크포인트를 지운다."""
        if not self.keep_final and self.path.exists():
            self.path.unlink()
            logger.info(f"[Resume] finished, removed {self.path}")
