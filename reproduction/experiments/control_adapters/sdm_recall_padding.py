"""State-neutral per-bank padding for native SDM on Adaptive Recall.

The pinned SDM kernel flattens bank/time before chunking. For T=272 or 528, an
unpadded C=256 chunk crosses independent banks; padding each bank with zero
gates and weights prevents that.
"""
import torch


def pad_parallel_time(tensors, *, chunk_size, rows_per_bank):
    banks, time = tensors[0].shape[:2]
    padding = (-time) % chunk_size
    if not padding:
        return tensors
    result = []
    for index, tensor in enumerate(tensors):
        shape = list(tensor.shape)
        shape[1] = padding
        fill = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
        if index in (0, 5):
            # Even inactive native gathers need a valid address in their own bank.
            base = torch.arange(banks,device=tensor.device,dtype=tensor.dtype).view(banks,1,1)*rows_per_bank
            fill = fill + base
            if rows_per_bank >= 4096:
                # Every padded gate/weight is zero. Spread these inactive
                # addresses so the native event-list kernel does not traverse
                # thousands of dummy events concentrated on one row.
                selected=tensor.shape[-1]
                offsets=torch.arange(padding*selected,device=tensor.device,dtype=tensor.dtype)
                fill=base+(offsets.reshape(1,padding,selected)%rows_per_bank)
        result.append(torch.cat((tensor,fill),dim=1))
    return tuple(result)

