"""The two adopted optimizer recipes, with explicit checkpointable FP32 masters."""

import math
import torch


class AdamW:
    def __init__(self, network, benchmark):
        self.model_parameters = list(network.parameters())
        self.master_parameters = ([torch.nn.Parameter(p.detach().float().clone()) for p in self.model_parameters]
                                  if benchmark == "wiki" else self.model_parameters)
        self.has_masters = benchmark == "wiki"
        self.benchmark = benchmark
        groups = [{"params": [p for p, m in zip(self.master_parameters, self.model_parameters, strict=True)
                              if bool(getattr(m, "_no_weight_decay", False)) == no_decay],
                   "weight_decay": 0.0 if no_decay else 0.01} for no_decay in (False, True)]
        self.optimizer = torch.optim.AdamW(groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                                           fused=self.model_parameters[0].is_cuda)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        for p in self.model_parameters:
            p.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, lr):
        for group in self.param_groups:
            group["lr"] = lr
        if self.benchmark == 'wiki':
            norm = torch.nn.utils.clip_grad_norm_(self.model_parameters, 1.0, error_if_nonfinite=True)
        else:
            # Preserve the final comparison's FP32 norm accumulation; Adam
            # parameters and moments remain BF16 under the Recall standard.
            grads = [p.grad for p in self.model_parameters if p.grad is not None]
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g, dtype=torch.float32) for g in grads]))
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite gradient norm')
            coefficient = (1. / (norm + 1e-6)).clamp(max=1.)
            by_dtype = {}
            for g in grads:
                by_dtype.setdefault(g.dtype, []).append(g)
            for group in by_dtype.values():
                torch._foreach_mul_(group, coefficient)
        if self.has_masters:
            for p, master in zip(self.model_parameters, self.master_parameters, strict=True):
                master.grad = None if p.grad is None else p.grad.float()
        self.optimizer.step()
        if self.has_masters:
            for p, master in zip(self.model_parameters, self.master_parameters, strict=True):
                p.copy_(master)
        return float(norm)

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(),
                "masters": [p.detach().cpu() for p in self.master_parameters] if self.has_masters else None}

    @torch.no_grad()
    def load_state_dict(self, state):
        if (state["masters"] is not None) != self.has_masters:
            raise ValueError("Optimizer precision changed")
        if self.has_masters:
            for p, saved in zip(self.master_parameters, state["masters"], strict=True):
                if saved.dtype != torch.float32 or p.shape != saved.shape:
                    raise ValueError("Invalid FP32 master checkpoint")
                p.copy_(saved)
        # These optimizer parameters are FP32 for Wiki and BF16 for Recall;
        # PyTorch's state restoration therefore preserves the correct moment dtype.
        self.optimizer.load_state_dict(state["optimizer"])
        for p, values in self.optimizer.state.items():
            for name in ("exp_avg", "exp_avg_sq"):
                if values[name].dtype != p.dtype:
                    raise ValueError("Optimizer moment precision changed")


def learning_rate(benchmark, step):
    if benchmark == "recall":
        return 3e-4 * min(step / 100, 1.0)
    if step <= 540:
        return 3e-4 * step / 540
    return 3e-4 * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - 540) / (21603 - 540))))
