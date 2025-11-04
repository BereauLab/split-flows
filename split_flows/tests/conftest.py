import pytest
import torch
import tempfile
import os
from torch import Tensor

from split_flows.mixins.continuous_flow import ContinuousFlowMixin
from split_flows.models.split_flow import (
    NoiseAugmentation,
    VelocityNet,
    SplitFlow,
    SplitFlowHparams,
)


class SimpleContinuousFlow(ContinuousFlowMixin):
    """Simple test implementation of ContinuousFlowMixin with linear velocity."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def velocity(self, xt: Tensor, t: Tensor) -> Tensor:
        """Linear velocity field: v(x,t) = x.

        :param xt: Position of the system at time t.
        :param t: Time.
        :return: Velocity of the system."""
        return xt


class ConstantVelocityFlow(ContinuousFlowMixin):
    """Test implementation with constant velocity field."""

    def __init__(self, velocity_value: float = 1.0):
        self.velocity_value = velocity_value

    def velocity(self, xt: Tensor, t: Tensor) -> Tensor:
        """Constant velocity field.

        :param xt: Position of the system at time t.
        :param t: Time.
        :return: Constant velocity."""
        return xt * 0 + torch.ones_like(xt) * self.velocity_value


@pytest.fixture
def simple_flow():
    """Create a simple continuous flow instance.

    :return: SimpleContinuousFlow instance."""
    return SimpleContinuousFlow(dim=2)


@pytest.fixture
def constant_flow():
    """Create a constant velocity flow instance.

    :return: ConstantVelocityFlow instance."""
    return ConstantVelocityFlow(velocity_value=1.0)


@pytest.fixture
def sample_batch():
    """Create a sample batch of 2D coordinates.

    :return: Tensor of shape (batch_size, dim)."""
    batch_size = 10
    dim = 2
    return torch.randn(batch_size, dim)


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing.

    :return: Tensor of shape (batch_size, time_steps, dim)."""
    batch_size = 5
    time_steps = 10
    dim = 3
    return torch.randn(batch_size, time_steps, dim)


@pytest.fixture
def latent_groupings():
    """Create sample latent groupings for testing.

    :return: List of tuples mapping CG indices to noise indices."""
    return [(0, [1, 2]), (3, [4, 5]), (6, [7, 8])]


@pytest.fixture
def noise_augmentation(latent_groupings):
    """Create a NoiseAugmentation instance.

    :param latent_groupings: Latent groupings fixture.
    :return: NoiseAugmentation instance."""
    return NoiseAugmentation(
        latent_groupings=latent_groupings,
        scale=0.1,
        num_particles=9,
    )


@pytest.fixture
def velocity_net():
    """Create a VelocityNet instance.

    :return: VelocityNet instance."""
    atom_types = torch.randn(9, 5)
    bead_types = torch.randn(9, 3)
    return VelocityNet(
        dim=2,
        depth=1,
        fourier_features=0,
        atom_types=atom_types,
        bead_types=bead_types,
    )


@pytest.fixture
def trained_velocity_net(velocity_net):
    """Create a VelocityNet with non-zero weights for testing.

    :param velocity_net: Base VelocityNet fixture.
    :return: VelocityNet with initialized weights."""
    # Initialize weights to be larger
    for param in velocity_net.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_normal_(param, gain=2.0)
    return velocity_net


@pytest.fixture
def sample_cg_coords():
    """Create sample coarse-grained coordinates.

    :return: Tensor of shape (batch_size, num_cg_beads, 3)."""
    batch_size = 4
    num_cg_beads = 3
    return torch.randn(batch_size, num_cg_beads, 3)


@pytest.fixture
def sample_aa_coords():
    """Create sample all-atom coordinates.

    :return: Tensor of shape (batch_size, num_particles, 3)."""
    batch_size = 4
    num_particles = 9
    return torch.randn(batch_size, num_particles, 3)


@pytest.fixture
def mock_topology_files():
    """Create mock topology files for testing.

    :return: Tuple of paths to AA and CG topology files."""
    aa_pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       0.000   1.000   0.000  1.00  0.00           C
ATOM      4  CA  ALA A   2       0.000   0.000   1.000  1.00  0.00           C
ATOM      5  CB  ALA A   2       1.000   0.000   1.000  1.00  0.00           C
ATOM      6  C   ALA A   2       0.000   1.000   1.000  1.00  0.00           C
ATOM      7  CA  ALA A   3       0.000   0.000   2.000  1.00  0.00           C
ATOM      8  CB  ALA A   3       1.000   0.000   2.000  1.00  0.00           C
ATOM      9  C   ALA A   3       0.000   1.000   2.000  1.00  0.00           C
END
"""
    cg_pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       0.000   0.000   1.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       0.000   0.000   2.000  1.00  0.00           C
END
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(aa_pdb)
        aa_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(cg_pdb)
        cg_path = f.name

    yield aa_path, cg_path

    os.unlink(aa_path)
    os.unlink(cg_path)


@pytest.fixture
def split_flow_hparams(mock_topology_files, latent_groupings):
    """Create SplitFlowHparams for testing.

    :param mock_topology_files: Mock topology files fixture.
    :param latent_groupings: Latent groupings fixture.
    :return: SplitFlowHparams instance."""
    aa_path, cg_path = mock_topology_files
    return SplitFlowHparams(
        aa_topology_path=aa_path,
        cg_topology_path=cg_path,
        atom_encoding_attr="name",
        bead_encoding_attr="resname",
        latent_noise=0.1,
        latent_groupings=latent_groupings,
        velo_net_hidden_dim=16,
        velo_net_depth=2,
        velo_net_fourier_features=8,
        sigma=0.0,
        interpolation="linear",
    )


@pytest.fixture
def split_flow(split_flow_hparams):
    """Create a SplitFlow model instance.

    :param split_flow_hparams: SplitFlowHparams fixture.
    :return: SplitFlow instance."""
    return SplitFlow(split_flow_hparams)


@pytest.fixture
def mock_trajectory_files():
    """Create mock trajectory and topology files for testing MDTrajectoryData.

    :return: Tuple of paths to trajectory and topology files."""
    import numpy as np
    import mdtraj

    # Create a simple topology with some atoms including hydrogens and water
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)

    # Add some heavy atoms
    ca_atom = topology.add_atom("CA", mdtraj.element.carbon, residue)
    cb_atom = topology.add_atom("CB", mdtraj.element.carbon, residue)
    c_atom = topology.add_atom("C", mdtraj.element.carbon, residue)

    # Add hydrogens
    h1_atom = topology.add_atom("H1", mdtraj.element.hydrogen, residue)
    h2_atom = topology.add_atom("H2", mdtraj.element.hydrogen, residue)

    # Add water molecules as solvent
    water_res = topology.add_residue("HOH", chain)
    o_atom = topology.add_atom("O", mdtraj.element.oxygen, water_res)
    h3_atom = topology.add_atom("H1", mdtraj.element.hydrogen, water_res)
    h4_atom = topology.add_atom("H2", mdtraj.element.hydrogen, water_res)

    # Create trajectory data with 10 frames
    n_frames = 100
    n_atoms = topology.n_atoms
    xyz = np.random.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.5

    # Create trajectory
    traj = mdtraj.Trajectory(xyz, topology)

    # Save to temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        traj[0].save_pdb(f.name)
        topology_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dcd", delete=False) as f:
        traj_path = f.name
        traj.save_dcd(traj_path)

    yield traj_path, topology_path

    os.unlink(traj_path)
    os.unlink(topology_path)


@pytest.fixture
def basic_md_hparams(mock_trajectory_files):
    """Create basic MDTrajectoryDataHparams for testing.

    :param mock_trajectory_files: Mock trajectory files fixture.
    :return: MDTrajectoryDataHparams instance."""
    from split_flows.data.md_trajectory_data import MDTrajectoryDataHparams

    traj_path, top_path = mock_trajectory_files
    return MDTrajectoryDataHparams(
        trajectory_path=traj_path,
        topology_path=top_path,
        remove_solvent=False,
        superpose=False,
        center_coordinates=False,
        prune_hydrogens=False,
        stride=1,
    )
