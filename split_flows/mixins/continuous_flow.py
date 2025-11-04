from typing import Literal, cast

from tqdm import tqdm

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torchdiffeq import odeint

from split_flows.utils.utils import gradient


class ContinuousFlowMixin(ABC):
    """Implementation of some basic continuous flow utilities."""

    @abstractmethod
    def velocity(self, xt: Tensor, t: Tensor) -> Tensor:
        """Computes the velocity of the flow at a given time.

        :param xt: position of the system at time t.
        :param t: time.
        :return: velocity of the system."""
        pass

    @torch.set_grad_enabled(True)
    def divergence(self, xt: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the divergence of the velocity field at (xt, t).

        :param xt: position of the system at time t.
        :param t: time.
        :return: velocity of the system and divergence of the velocity field."""

        xt_flat = xt.flatten(1)
        xt_flat.requires_grad_()
        vt = self.velocity(xt_flat.view(xt.shape), t)
        vt_flat = vt.flatten(1)

        div_vt = torch.zeros(xt.shape[0], device=xt.device)
        for i in range(vt_flat.shape[1]):
            div_vt += gradient(vt_flat[:, i], xt_flat, create_graph=True)[:, i]

        return vt, div_vt

    @torch.no_grad()
    def compute_volume_change(
        self,
        x0: Tensor,
        chunk_size: int | None = None,
        method: Literal["euler", "rk4"] = "euler",
        step_size: float = 1e-1,
        return_intermediate: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Compute the volume change integrating the flow from (x0, 0) to (x1, 1).

        :param x0: Starting coordinates of the system.
        :param chunk_size: Optional chunk size for processing large batch sizes.
        :return: End point sample and volume change per sample: (B,)"""

        def ode_func(t, y):
            vt, divt = self.divergence(y[0], t)
            return (vt.detach(), divt.detach())

        if chunk_size is None:
            chunk_size = int(x0.shape[0])

        time_grid = torch.linspace(0, 1, int(1 / step_size) + 1, device=x0.device)
        div0 = torch.zeros(x0.shape[0], device=x0.device)

        sol = torch.zeros((time_grid.shape[0], *x0.shape), device=x0.device)
        volume_change = torch.zeros((time_grid.shape[0], x0.shape[0]), device=x0.device)
        for i in tqdm(range(0, x0.shape[0], chunk_size)):
            end = min(i + chunk_size, x0.shape[0])
            sol_i, volume_change_i = cast(
                Tensor,
                odeint(
                    ode_func,
                    (x0[i:end], div0[i:end]),
                    time_grid,
                    method=method,
                    options={"step_size": step_size},
                ),
            )
            sol[:, i:end] = sol_i
            volume_change[:, i:end] = volume_change_i

        volume_change *= -1

        if return_intermediate:
            return sol, volume_change

        return sol[-1], volume_change[-1]

    def compute_flow(
        self,
        x0: Tensor,
        chunk_size: int | None = None,
        method: Literal["euler", "rk4"] = "euler",
        step_size: float = 1e-1,
        reverse: bool = False,
    ) -> Tensor:
        """Compute path samples using the ODE solver starting from the given coordinates. Returns
        samples of the path at the specified time grid.

        :param x0: Starting coordinates of the system.
        :param chunk_size: Optional chunk size for processing large batch sizes.
        :param method: ODE solver method to use.
        :param step_size: Step size for the ODE solver.
        :param reverse: Whether to compute the flow in reverse (from x1 to x0).
        :return: Path samples at the specified time grid."""

        time_grid = torch.linspace(0, 1, int(1 / step_size) + 1, device=x0.device)

        def ode_func(t, y):
            return self.velocity(y, t)

        if reverse:
            time_grid = torch.flip(time_grid, dims=(0,))

        if chunk_size is None:
            chunk_size = int(x0.shape[0])

        sol = torch.zeros((time_grid.shape[0], *x0.shape), device=x0.device)
        for i in tqdm(range(0, x0.shape[0], chunk_size)):
            end = min(i + chunk_size, x0.shape[0])
            sol_i = cast(
                Tensor,
                odeint(
                    ode_func,
                    x0[i:end],
                    time_grid,
                    method=method,
                    options={"step_size": step_size},
                ),
            )
            sol[:, i:end] = sol_i

        return sol
