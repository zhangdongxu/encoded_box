import torch
from torch.nn import Module
import math


def printing(metrics, prefix=""):
    print_info = f"{prefix}: Average Precision: {metrics['AP']:>3f}\n" 
    print(print_info)


def log1mexp(
    x: torch.Tensor,
    split_point: float = math.log(0.5),
    exp_zero_eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes log(1 - exp(x)).
    Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).
    = log1p(-exp(x)) when x <= log(1/2)
    or
    = log(-expm1(x)) when log(1/2) < x <= 0
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
    Args:
        x: input tensor
        split_point: Should be kept to the default of log(0.5)
        exp_zero_eps: Default 1e-7
    Returns:
        torch.Tensor: Elementwise log1mexp(x) = log(1-exp(x))
    """
    logexpm1_switch = x > split_point
    Z = torch.zeros_like(x)
    # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
    # ie. p=1
    logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
    # hack the backward pass
    # if expm1(x) gets very close to zero, then the grad log() will produce inf
    # and inf*0 = nan. Hence clip the grad so that it does not produce inf
    logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
    Z[logexpm1_switch] = logexpm1.detach() + (
        logexpm1_bw - logexpm1_bw.detach()
    )
    # Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
    Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

    return Z


class BCEWithLogsLoss(Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        :param input: log probabilities
        :param target: target probabilities
        :param weights: weight on each data point
        """
        return ((-(target * input + (1 - target) * log1mexp(input))) * weights).sum() / weights.sum()

