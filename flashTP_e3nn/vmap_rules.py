from typing import Sequence

import torch


def make_loop_vmap_rule(op_name: str):
    # torch.func.vmap support: call the op once per batch element and stack
    namespace, name = op_name.split("::")

    def loop_vmap_rule(info, in_dims, *args):
        op = getattr(getattr(torch.ops, namespace), name)
        per_element = []
        for index in range(info.batch_size):
            # non-tensor args (ints, List[int]) arrive with in_dim None
            sliced = [
                arg.select(dim, index)
                if isinstance(arg, torch.Tensor) and isinstance(dim, int)
                else arg
                for arg, dim in zip(args, in_dims)
            ]
            per_element.append(op(*sliced))
        if isinstance(per_element[0], torch.Tensor):
            return torch.stack(per_element), 0
        stacked = [
            torch.stack([outs[k] for outs in per_element])
            for k in range(len(per_element[0]))
        ]
        return stacked, [0] * len(stacked)

    return loop_vmap_rule


def register_loop_vmap_rules(op_names: Sequence[str]):
    for op_name in op_names:
        torch.library.register_vmap(op_name, make_loop_vmap_rule(op_name))
