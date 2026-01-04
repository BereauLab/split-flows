"""Core split-flow model utilities.

Contains NoiseAugmentation, VelocityNet and the SplitFlow Model.
"""

from typing import cast
import logging

from math import sqrt, pi

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scienceplots  # noqa: F401
import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Distribution

from egnn_pytorch import EGNN_Network
from sklearn.mixture import GaussianMixture
from hydrantic.model import Model, ModelHparams
from moleculekit.molecule import Molecule

from split_flows.utils.interpolant import Interpolation, Interpolant
from split_flows.mixins.continuous_flow import ContinuousFlowMixin
from split_flows.utils.utils import to_one_hot, sum_except_batch, match_dims


logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("SplitFlow")


class NoiseAugmentation(Distribution):
    """Noise augmentation Distribution for mapping CG coords to full coords.

    Provides augment, augment_gmm, to_standard_normal, log_prob, energy, and force.
    """

    _validate_args = False

    def __init__(
        self,
        latent_groupings: list[tuple[int, list[int]]],
        scale: float,
        num_particles: int,
    ):
        super().__init__()
        self.latent_groupings = latent_groupings
        self.scale = scale
        self.num_particles = num_particles

    def augment(self, R: Tensor, temperature: float = 1.0) -> Tensor:
        """Augment the coarse-grained coordinates with noise to create the full set of coordinates.

        :param R: Coarse-grained coordinates.
        :param temperature: Temperature scaling factor for the noise.
        :return: Full set of coordinates with noise."""

        z = torch.empty((R.shape[0], self.num_particles, R.shape[2]), device=R.device)
        for i, (cg_idx, noise_idx) in enumerate(self.latent_groupings):
            z[:, cg_idx, :] = R[:, i, :]
            z[:, noise_idx, :] = R[:, i, :][
                :, None, :
            ] + temperature * self.scale * torch.randn_like(z[:, noise_idx, :])

        return z

    def augment_gmm(self, R: Tensor, gmm: GaussianMixture) -> Tensor:
        """Augment the coarse-grained coordinates with noise sampled from a Gaussian mixture
        model to create the full set of coordinates.

        :param R: Coarse-grained coordinates.
        :param gmm: Gaussian mixture model to sample noise from.
        :return: Full set of coordinates with noise."""

        z = torch.empty((R.shape[0], self.num_particles, R.shape[2]), device=R.device)
        z_gmm = torch.tensor(gmm.sample(R.shape[0])[0], dtype=R.dtype, device=R.device).view(
            R.shape[0], -1, 3
        )

        start_idx = 0
        for i, (cg_idx, noise_idx) in enumerate(self.latent_groupings):
            z[:, cg_idx, :] = R[:, i, :]
            z[:, noise_idx, :] = (
                R[:, i, :][:, None, :]
                + self.scale * z_gmm[:, start_idx : start_idx + len(noise_idx), :]
            )
            start_idx += len(noise_idx)

        return z

    def to_standard_normal(self, R_eps: Tensor) -> Tensor:
        """Center and rescale the noise coordinates.

        :param R_eps: Coarse-grained coordinates with noise.
        :return: Whitened noise coordinates."""

        R_whitened = torch.zeros_like(R_eps, device=R_eps.device)
        for cg_idx, noise_idx in self.latent_groupings:
            R_cg = R_eps[:, cg_idx, :]
            R_noise = R_eps[:, noise_idx, :]
            R_whitened[:, cg_idx, :] = R_cg
            R_whitened[:, noise_idx, :] = (R_noise - R_cg[:, None, :]) / self.scale

        return R_whitened

    def log_prob(self, value: Tensor) -> Tensor:
        """Compute the log probability of the noise augmentation.

        :param value: Coarse-grained coordinates with noise.
        :return: Log probability of the noise coordinates."""

        Z = torch.tensor([1 / (self.scale * sqrt(2 * pi))], device=value.device)

        log_prob = torch.zeros(value.shape[0], device=value.device)
        for cg_idx, noise_idx in self.latent_groupings:
            R_cg = value[:, cg_idx, :]
            R_noise = value[:, noise_idx, :]
            exponential_term = (
                -0.5 * sum_except_batch((R_noise - R_cg[:, None, :]) ** 2) / self.scale**2
            )
            normalization_term = -torch.log(Z) * R_noise.shape[1]
            log_prob += exponential_term + normalization_term

        return log_prob


class VelocityNet(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        fourier_features: int,
        atom_types: Tensor,
        bead_types: Tensor,
    ):
        super(VelocityNet, self).__init__()
        self.dim = dim
        self.depth = depth
        self.fourier_features = fourier_features
        self.register_buffer("atom_types", atom_types.unsqueeze(0))
        self.register_buffer("bead_types", bead_types.unsqueeze(0))

        self.net = EGNN_Network(
            dim=2 * self.dim + 1,
            depth=self.depth,
            fourier_features=self.fourier_features,
        )

        self.atom_embedding = nn.Linear(atom_types.size(-1), self.dim)
        self.bead_embedding = nn.Linear(bead_types.size(-1), self.dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        atom_embeddings = self.atom_embedding(self.atom_types).repeat(x.size(0), 1, 1)
        bead_embeddings = self.bead_embedding(self.bead_types).repeat(x.size(0), 1, 1)
        t = match_dims(t, x).repeat(1, x.shape[1], 1)
        h = torch.cat([atom_embeddings, bead_embeddings, t], dim=-1)
        return self.net(h + torch.randn_like(h), x)[1]


class SplitFlowHparams(ModelHparams):
    aa_topology_path: str
    cg_topology_path: str

    cg_map_matrix_path: str | None = None

    atom_encoding_attr: str = "name"
    bead_encoding_attr: str = "resname"

    latent_noise: float
    latent_groupings: list[tuple[int, list[int]]]

    velo_net_hidden_dim: int
    velo_net_depth: int
    velo_net_fourier_features: int

    sigma: float = 0.0
    interpolation: Interpolation = "linear"


class SplitFlow(Model[SplitFlowHparams], ContinuousFlowMixin):
    hparams_schema = SplitFlowHparams

    def __init__(self, thparams: SplitFlowHparams):
        super(SplitFlow, self).__init__(thparams)

        # Load the all-atom and coarse-grained topologies
        self.mol_aa = Molecule(self.thparams.aa_topology_path)
        self.mol_cg = Molecule(self.thparams.cg_topology_path)

        # Define the CG mapping
        if not hasattr(self.thparams, "cg_map_matrix_path"):
            self.thparams.cg_map_matrix_path = None

        if self.thparams.cg_map_matrix_path is not None:
            console_logger.info("Using provided CG map matrix.")
            self.register_buffer(
                "map_matrix",
                torch.load(self.thparams.cg_map_matrix_path, weights_only=True),
            )
            self.cg_map = lambda x: self.map_matrix.T @ x
        else:
            console_logger.info("Using index-based CG map.")
            self.cg_map = lambda x: x[:, self.indices_split[0], :]

        # One-hot encode atom and bead types per particle
        if getattr(self.mol_aa, self.thparams.atom_encoding_attr) is None:
            raise ValueError(
                f"Atom encoding attribute '{self.thparams.atom_encoding_attr}' \
                    not found in all-atom molecule."
            )
        if getattr(self.mol_aa, self.thparams.bead_encoding_attr) is None:
            raise ValueError(
                f"Bead encoding attribute '{self.thparams.bead_encoding_attr}' \
                    not found in all-atom molecule."
            )
        self.register_buffer(
            "atom_types",
            to_one_hot(
                getattr(self.mol_aa, self.thparams.atom_encoding_attr),
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self.register_buffer(
            "bead_types",
            to_one_hot(
                getattr(self.mol_aa, self.thparams.bead_encoding_attr),
                device=self.device,
                dtype=self.dtype,
            ),
        )

        # Noise augmentation
        self.noise = NoiseAugmentation(
            latent_groupings=self.thparams.latent_groupings,
            scale=self.thparams.latent_noise,
            num_particles=self.num_particles,
        )

        # Velocity network
        self.velo_net = VelocityNet(
            dim=self.thparams.velo_net_hidden_dim,
            depth=self.thparams.velo_net_depth,
            fourier_features=self.thparams.velo_net_fourier_features,
            atom_types=cast(Tensor, self.atom_types),
            bead_types=cast(Tensor, self.bead_types),
        )

        # Interpolant
        self.interpolant = Interpolant(
            interpolation=self.thparams.interpolation,
            sigma=self.thparams.sigma,
        )

    def velocity(self, xt: Tensor, t: Tensor) -> Tensor:
        """Compute the velocity at position xt and time t.

        :param xt: Coordinates at time t.
        :param t: Time at which to compute the velocity.
        :return: Velocity."""

        t = match_dims(t, xt)
        if t.size(0) != xt.size(0):
            t = t.repeat(xt.size(0), 1, 1)

        return self.velo_net(xt, t)

    def compute_metrics(self, batch: tuple[Tensor, ...], batch_idx: int) -> dict[str, Tensor]:
        """Compute training/validation metrics.

        :param batch: Batch data tuple, expecting (r,) where r is a Tensor.
        :param batch_idx: Index of the batch.
        :return: Dictionary of scalar tensors for logging."""

        metrics: dict[str, Tensor] = {}

        metrics["loss"] = torch.tensor(0.0, device=self.device)
        (r,) = batch
        R = self.cg_map(r)
        x0 = self.noise.augment(R)
        x1 = r

        t = torch.rand(x0.shape[0], 1, 1, device=x0.device)
        xt, vt, _ = self.interpolant.compute_path_sample(t, x0, x1)
        vt_hat = self.velocity(xt, t)

        # flow matching loss
        metrics["loss_fm"] = sum_except_batch(torch.pow(vt_hat - vt, 2)).mean()
        metrics["loss"] += metrics["loss_fm"]

        return metrics

    @property
    def num_particles(self) -> int:
        """Return the number of particles in the system.

        :return: Number of particles in the system."""

        num_particles = 0
        for _, noise_idx in self.thparams.latent_groupings:
            num_particles += len(noise_idx) + 1
        return num_particles

    @property
    def indices_split(self) -> tuple[Tensor, Tensor]:
        """Return the coarse-grained and noise particle indices as tensors.

        :returns: A tuple (cg_indices, noise_indices) with dtype torch.long on self.device."""
        cg_indices = torch.tensor(
            [index for index, _ in self.thparams.latent_groupings],
            device=self.device,
            dtype=torch.long,
        )

        noise_list: list[int] = []
        for _, noise_indices in self.thparams.latent_groupings:
            noise_list.extend(noise_indices)

        noise_indices = torch.tensor(noise_list, device=self.device, dtype=torch.long)

        return cg_indices, noise_indices
