"""FP32 master AdamW, adapted from the established WikiText optimizer.

Source: .deps/m5/benchmarks/train_routing_capacity_wikitext_cuda.py.
Keep master weights before casting the compute model to BF16. Accumulate
microbatch gradients and clip in FP32; serialize every master and moment.
"""
from __future__ import annotations

import torch
from torch import nn


class FP32MasterAdamW:
    def __init__(self, model: nn.Module, *, lr: float, betas: tuple[float, float],
                 weight_decay: float, eps: float = 1e-8, honor_no_decay: bool = True,
                 fused: bool | None = None):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in named]
        self.model_parameters = [p for _, p in named]
        self.master_parameters = [nn.Parameter(p.detach().float().clone()) for _, p in named]
        groups = []
        for excluded in (False, True):
            values = [q for p, q in zip(self.model_parameters, self.master_parameters, strict=True)
                      if bool(honor_no_decay and getattr(p, "_no_weight_decay", False)) == excluded]
            if values:
                groups.append({"params": values, "weight_decay": 0.0 if excluded else weight_decay})
        if fused is None:
            fused = self.master_parameters[0].is_cuda
        self.optimizer = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)
        self._accumulated = False

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, *, set_to_none=True):
        if not set_to_none:
            raise ValueError("this campaign requires set_to_none=True")
        for p in self.model_parameters:
            p.grad = None
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulated = False

    @torch.no_grad()
    def accumulate(self):
        for p, master in zip(self.model_parameters, self.master_parameters, strict=True):
            if p.grad is not None:
                gradient = p.grad.detach().float()
                if master.grad is None:
                    master.grad = gradient.clone()
                else:
                    master.grad.add_(gradient)
                p.grad = None
        self._accumulated = True

    @torch.no_grad()
    def clip_grad_norm(self, maximum: float):
        if not self._accumulated:
            self.accumulate()
        return torch.nn.utils.clip_grad_norm_(self.master_parameters, maximum, error_if_nonfinite=True)

    @torch.no_grad()
    def step(self):
        if not self._accumulated:
            self.accumulate()
        self.optimizer.step()
        self.sync_to_model()

    @torch.no_grad()
    def sync_to_model(self):
        for p, master in zip(self.model_parameters, self.master_parameters, strict=True):
            p.copy_(master)

    def state_dict(self):
        return {"schema_version": 2, "names": self.names,
                "masters": [p.detach().cpu().clone() for p in self.master_parameters],
                "optimizer": self.optimizer.state_dict()}

    @torch.no_grad()
    def load_state_dict(self, state):
        if state.get("schema_version") != 2 or state.get("names") != self.names:
            raise ValueError("optimizer identity/order changed")
        if len(state["masters"]) != len(self.master_parameters):
            raise ValueError("missing master parameters")
        for p, v in zip(self.master_parameters, state["masters"], strict=True):
            if p.shape != v.shape or v.dtype != torch.float32:
                raise ValueError("master parameter shape/precision changed")
            p.copy_(v)
        self.optimizer.load_state_dict(state["optimizer"])
        self.assert_precision()
        self.sync_to_model()
        self.zero_grad()

    def assert_precision(self):
        if any(p.dtype != torch.float32 for p in self.master_parameters):
            raise ValueError("master weights must remain FP32")
        for p, state in self.optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in state or state[key].dtype != torch.float32 or state[key].shape != p.shape:
                    raise ValueError(f"invalid FP32 optimizer state: {key}")

    def master_model_state(self, model):
        parameter_names = {n for n, _ in model.named_parameters(remove_duplicate=False)}
        state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n not in parameter_names}
        # Include all aliases of tied parameters in the model state dict.
        masters = {id(p): q for p, q in zip(self.model_parameters, self.master_parameters, strict=True)}
        cpu_masters = {key: p.detach().cpu().clone() for key, p in masters.items()}
        for n, p in model.named_parameters(remove_duplicate=False):
            if id(p) in masters:
                state[n] = cpu_masters[id(p)]
            else:
                state[n] = p.detach().cpu().clone()
        return state
