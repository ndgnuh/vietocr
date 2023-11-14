from dataclasses import dataclass


def acc_full_sequence(pr: str, gt: str) -> float:
    """Compute full-sequence accuracy, returns 1 if `pr` and `gt` equals, else 0.

    Args:
        pr (str): prediction
        gt (str): ground truth
    """
    return 1.0 * (pr == gt)


def acc_per_char(pr: str, gt: str) -> float:
    """Compute per-characters accuracy.

    In case ground truth is empty, return 1 if `pr` is also empty, else return 0.
    In case `gt` is not empty, return the ratio `correct_count / len(gt)`, where
    `correct_count` is the number of characters where `pr[i] == gt[i]`.

    Args:
        pr (str): prediction
        gt (str): ground truth
    """
    gt_len = len(gt)
    pr_len = len(gt)

    # Special case
    if gt_len == 0:
        return 1 * (pr_len == 0)

    # Number of matching character
    correct_count = sum((c_pr == c_gt) for (c_pr, c_gt) in zip(pr, gt))
    return correct_count / gt_len


def acc_fuzzy(pr: str, gt: str) -> float:
    """Fuzzy ratio accuracy, require thefuzz package.

    Args:
        pr (str): prediction
        gt (str): ground truth
    """
    from thefuzz import ratio

    return ratio(pr, gt) / 100


@dataclass
class Avg:
    acc: float = 0
    count: int = 0

    def get(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.acc / self.count
        return mean

    def reset(self):
        self.acc = 0
        self.count = 0
        return self.mean

    def append(self, x):
        self.acc += x
        self.count += 1
