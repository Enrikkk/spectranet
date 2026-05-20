"""Loss functions: relative L^p loss and a Sobolev (H^s) variant.

Verbatim ports of the loss classes used in the original trainer; kept under
their original names for cross-codebase compatibility.
"""

from __future__ import annotations
import torch


class LpLoss(object):
    """Relative L^p loss — the canonical neural-operator metric.

    With defaults ``d=2, p=2``, ``LpLoss()(x, y)`` returns
    ``mean_b ‖x − y‖_2 / ‖y‖_2`` (the ``rel`` method).  ``size_average=False``
    returns the per-sample sum instead of the mean (used inside the AR
    training loop to avoid double-averaging).
    """

    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True):
        super().__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            return (
                torch.mean(diff_norms / y_norms)
                if self.size_average
                else torch.sum(diff_norms / y_norms)
            )
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class HsLoss(object):
    """Relative H^s (Sobolev) loss — penalises mismatched derivatives in addition
    to the L² mismatch.  Not used by the SpectraNet; preserved here
    so that ablation runs that rely on it remain reproducible."""

    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        k: int = 1,
        a=None,
        group: bool = False,
        size_average: bool = True,
        reduction: bool = True,
    ):
        super().__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average
        self.a = [1] * k if a is None else a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            return (
                torch.mean(diff_norms / y_norms)
                if self.size_average
                else torch.sum(diff_norms / y_norms)
            )
        return diff_norms / y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat(
            (
                torch.arange(start=0, end=nx // 2, step=1),
                torch.arange(start=-nx // 2, end=0, step=1),
            ),
            0,
        ).reshape(nx, 1).repeat(1, ny)
        k_y = torch.cat(
            (
                torch.arange(start=0, end=ny // 2, step=1),
                torch.arange(start=-ny // 2, end=0, step=1),
            ),
            0,
        ).reshape(1, ny).repeat(nx, 1)
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if not balanced:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x ** 2 + k_y ** 2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x ** 2 + k_y ** 2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss
