from typing import List
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class ADOPT(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.5, 0.99),
        eps=1e-6,
        weight_decay=0,
        decoupled=False,
        bias_correction=(False, True),
        *,
        maximize: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled=decoupled,
            bias_correction=bias_correction,
            maximize=maximize,
        )
        super(ADOPT, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ADOPT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("decoupled", False)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            bc1, bc2 = group["bias_correction"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("ADOPT does not support sparse gradients")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        # state["exp_avg_sq"] = torch.zeros_like(
                        #     p, memory_format=torch.preserve_format
                        # )
                        state["exp_avg_sq"] = torch.ones_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            adopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                bc1=bc1,
                bc2=bc2,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                decoupled=group["decoupled"],
                eps=group["eps"],
                maximize=group["maximize"],
            )
        return loss


def adopt(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    bc1: bool,
    bc2: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    decoupled: bool,
    eps: float,
    maximize: bool
):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            if decoupled:
                param.add_(param, alpha=-lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)


        denom = exp_avg_sq.add(eps**2).sqrt_()
        exp_avg.mul_(beta1).add_(grad.div(denom), alpha=1 - beta1)
        param.add_(exp_avg, alpha=-lr)

        # if step > 1:
        #     if bc2:
        #         denom = exp_avg_sq.add(eps**2)
        #         denom.div_(1 - beta2 ** (step - 1))
        #     else:
        #         denom = exp_avg_sq.add(eps**2)

        #     denom.sqrt_()
        #     exp_avg.mul_(beta1).add_(grad.div(denom), alpha=1 - beta1)

        #     if bc1:
        #         step_size = lr / (1 - beta1 ** (step - 1))
        #         param.add_(exp_avg, alpha=-step_size)
        #     else:
        #         param.add_(exp_avg, alpha=-lr)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
