from typing import Literal, Callable
import logging

import torch
from torch import Tensor

from split_flows.utils.utils import expand_like


logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("Interpolant")


Interpolation = Literal["linear", "trigonometric"]


class Interpolant:
    _alpha: Callable[[Tensor], Tensor]
    _alpha_dot: Callable[[Tensor], Tensor]
    _beta: Callable[[Tensor], Tensor]
    _beta_dot: Callable[[Tensor], Tensor]
    _gamma: Callable[[Tensor], Tensor]
    _gamma_dot: Callable[[Tensor], Tensor]

    def __init__(self, interpolation: Interpolation, sigma: float):
        self.interpolation = interpolation
        self.sigma = sigma

        self._configure_interpolation()

    def _configure_interpolation(self) -> None:
        match self.interpolation:
            case "linear":
                console_logger.info("Using a linear interpolation")
                self._alpha = lambda t: 1 - t
                self._alpha_dot = lambda t: -torch.ones_like(t)
                self._beta = lambda t: t
                self._beta_dot = lambda t: torch.ones_like(t)
            case "trigonometric":
                console_logger.info("Using a trigonometric interpolation")
                self._alpha = lambda t: torch.cos(torch.pi / 2 * t)
                self._alpha_dot = lambda t: -torch.pi / 2 * torch.sin(torch.pi / 2 * t)
                self._beta = lambda t: torch.sin(torch.pi / 2 * t)
                self._beta_dot = lambda t: torch.pi / 2 * torch.cos(torch.pi / 2 * t)
            case _:
                raise ValueError(f"Interpolation '{self.interpolation}' not supported")

        self._gamma = lambda t: (t * (1 - t)) * self.sigma
        self._gamma_dot = lambda t: (1 - 2 * t) * self.sigma

    def compute_path_sample(
        self, t: Tensor, x0: Tensor, x1: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Computes a sample from the interpolation path between x0 and x1 at time t.

        :param t: time.
        :param x0: initial position of the system.
        :param x1: final position of the system.
        :return: sampled position, reference velocity, and denoiser."""

        z = torch.randn_like(x0)
        It = self.compute_It(t, x0, x1, z)
        It_dot = self.compute_It_dot(t, x0, x1, z)
        return It, It_dot, -z

    def compute_It(self, t: Tensor, x0: Tensor, x1: Tensor, z: Tensor) -> Tensor:
        """Computes the interpolated position at time t.

        :param t: time.
        :param x0: initial position of the system.
        :param x1: final position of the system.
        :param z: noise vector.
        :return: interpolated position at time t."""

        alpha_t = expand_like(self._alpha(t), x0)
        beta_t = expand_like(self._beta(t), x1)
        gamma_t = expand_like(self._gamma(t), z)
        return alpha_t * x0 + beta_t * x1 + gamma_t * z

    def compute_It_dot(self, t: Tensor, x0: Tensor, x1: Tensor, z: Tensor) -> Tensor:
        """Computes the reference velocity at time t.

        :param t: time.
        :param x0: initial position of the system.
        :param x1: final position of the system.
        :param z: noise vector.
        :return: reference velocity at time t."""

        alpha_t_dot = expand_like(self._alpha_dot(t), x0)
        beta_t_dot = expand_like(self._beta_dot(t), x1)
        gamma_t_dot = expand_like(self._gamma_dot(t), z)
        return alpha_t_dot * x0 + beta_t_dot * x1 + gamma_t_dot * z
