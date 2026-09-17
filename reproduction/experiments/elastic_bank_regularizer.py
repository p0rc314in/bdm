"""Elastic SDM's final-union loss, using banks as addresses.

Same hard forward / noisy-OR straight-through gradient as elastic-sdm/sdm.py.
Reduce the sparse events directly instead of materializing [B,T,N].
No routing, recurrence, inference, or model parameter changes.
"""
import torch


def bank_occupancy(weights, indices, banks, valid=None):
    # Canonical top-k has distinct indices within each token. Collapsing time
    # before scatter therefore gives the same noisy OR as the released code.
    batch = weights.shape[0]
    addresses = indices.reshape(batch, -1)
    event_mask = torch.ones_like(indices, dtype=torch.int32)
    if valid is not None:
        event_mask = valid[..., None].expand_as(indices).to(torch.int32)
    log_survival = torch.log1p(-weights.float().clamp(0, 1 - 1e-6)) * event_mask
    totals = torch.zeros(batch, banks, device=weights.device, dtype=torch.float32)
    totals = totals.scatter_add(1, addresses, log_survival.reshape(batch, -1))
    soft = -torch.expm1(totals).mean(-1)
    counts = torch.zeros(batch, banks, device=weights.device, dtype=torch.int32)
    counts.scatter_add_(1, addresses, event_mask.reshape(batch, -1))
    hard = (counts > 0).float().mean(-1)
    return hard.detach() - soft.detach() + soft, hard, soft


def capture_write_route(layer):
    """Observe the existing first (write) route call; never recompute routing."""
    original = layer._route
    layer._elastic_capture = None

    def route(projected, selected):
        weights, indices = original(projected, selected)
        if layer._elastic_capture is None:
            layer._elastic_capture = bank_occupancy(weights, indices, layer.bank_count)
        return weights, indices

    layer._route = route
    return original
