import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from torch import optim
from torch.optim import lr_scheduler


@dataclass
class CosineDecayFunc:
    """A cosine function, decay from (x1, y1) to (x2, y2) in half pi.

    f(x) = 0.5(y1 - y2) * cos[(t - x1) / (x2 - x1)] + 0.5(y1 + y2)
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        self.T = math.pi / (self.x2 - self.x1)
        self.A = (self.y1 - self.y2) / 2
        self.C = (self.y1 + self.y2) / 2

    def __call__(self, x):
        return self.A * math.cos((x - self.x1) * self.T) + self.C


class CosineCycle:
    """
    A cycle with two cosine decay fn, one warmup and one decay.

    Args:
        n (float): Cycle length
        warmup (float): Number of warmup time, must be less than `n`
        ymin (float): Start `y` value
        ymax (float): Cycle peak `y` value
    """

    def __init__(self, n: float, warmup: float, ymin: float, ymax: float):
        assert warmup < n
        self.thresh = warmup
        self.cos_warmup = CosineDecayFunc(0, ymin, self.thresh, ymax)
        self.cos_decay = CosineDecayFunc(self.thresh, ymax, n, ymin)

    def __call__(self, t):
        if t <= self.thresh:
            return self.cos_warmup(t)
        else:
            return self.cos_decay(t)


@dataclass
class FunctionPieces:
    functions: List[Callable]
    length: int

    def __call__(self, x: float):
        idx = int(x / self.length)
        try:
            fn = self.functions[idx]
        except IndexError:
            fn = self.functions[-1]
        return fn(x - idx * self.length)


def CosineWWRD(
    optimizer: optim.Optimizer,
    total_steps: int,
    num_warmup_steps: int,
    cycle_length: Optional[int] = None,
    num_cycles: Optional[int] = None,
    peak_ratio: float = 1,
    start_ratio: float = 0.001,
    decay: float = 1,
):
    """Cosine annealing with Warmup, Restart and Decay.

    When using this function, either `num_cycles` or `cycle_length` must be provided,
    but not both.

    Args:
        optimizer (optim.Optimizer): torch's optimizer
        total_steps (int):
            Total number of stepping batches,
            will be used to determine number of cycles or cycle's length
        num_warmup_steps (int): Number of warmup steps for each cycle
        cycle_length (Optional[int]): Number of steps in each cycle
        num_cycles (Optional[int]): Number of cycles
        peak_ratio (float): Cycle peak value (default: 1)
        start_ratio (float): Ratio value at the start of the cycle (default: 1e-3)
        decay (float):
            Max ratio decay after each cycle, after each cycle,
            peak ratio is multiplied by this value (default is 1,
            which does nothing)

    """
    msg = "Either num cycles or cycle length must be provided"
    assert cycle_length is not None or num_cycles is not None, msg
    msg = "Can only provide num cycles OR cycle length"
    assert cycle_length is None or num_cycles is None, msg

    # Infer num cycles/cycle length
    if num_cycles is not None:
        cycle_length = total_steps / num_cycles
    else:
        num_cycles = int(math.ceil(total_steps / cycle_length))

    # Create ratio function
    cycles = [
        CosineCycle(
            cycle_length, num_warmup_steps, start_ratio, peak_ratio * decay**i
        )
        for i in range(num_cycles)
    ]
    fn = FunctionPieces(cycles, cycle_length)
    return lr_scheduler.LambdaLR(optimizer, fn)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from torch import nn
    from tqdm import tqdm

    model = nn.Linear(10, 10)
    total_steps = 3000
    cycle_length = 3000 // 3
    decay = 0.9
    warmup = 30
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    lr_scheduler = CosineWWRD(
        optimizer,
        total_steps=total_steps,
        cycle_length=cycle_length,
        num_warmup_steps=warmup,
        decay=decay,
    )

    lrs = []
    x = list(range(total_steps))
    for _ in tqdm(x):
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]
        lrs.append(lr)

    plt.plot(x, lrs)
    plt.show()

    # fn = FunctionPieces(
    #     [CosineCycle(1000, 200, 0.1, 0.3 * 0.99**i) for i in range(10)],
    #     1000,
    # )
    # x = np.arange(10 * 1000)
    # y = [fn(x_i) for x_i in x]
    # plt.plot(x, y)
    # plt.show()
