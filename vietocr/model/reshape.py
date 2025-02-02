# Generated by dirtytorch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, expr: str, mode="reshape"):
        super().__init__()
        assert mode in ["reshape", "view"]
        self.mode = mode
        expr = expr.replace(" ", "").replace(",", ", ")
        self.arg_expr, self.ret_expr = expr.split("->")
        exec(f"self.get_new_shape = lambda {self.arg_expr}: ({self.ret_expr})")

    def extra_repr(self):
        args = [
            f"[{self.arg_expr}] -> [{self.ret_expr}]".upper(),
            f"mode={self.mode}",
        ]
        return ", ".join(args)

    def forward(self, x):
        shape = self.get_new_shape(*x.shape)
        return getattr(x, self.mode)(*shape)


class Permute(nn.Module):
    def __init__(self, expr: str):
        super().__init__()

        expr = expr.replace(" ", "").replace(",", ", ")
        src_expr, dst_expr = expr.split("->")
        src = src_expr.split(", ")
        dst = dst_expr.split(", ")
        assert len(src) == len(dst)
        self.order = tuple([src.index(d) for d in dst])
        self.src_expr = src_expr
        self.dst_expr = dst_expr

    def extra_repr(self):
        return f"[{self.src_expr}] -> [{self.dst_expr}]".upper()

    def forward(self, x):
        return x.permute(self.order)
