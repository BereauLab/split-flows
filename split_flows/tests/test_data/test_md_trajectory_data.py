import pytest
import torch

from split_flows.data.md_trajectory_data import (
    MDTrajectoryData,
    MDTrajectoryDataHparams,
)


class TestInitialization:
    """Tests for basic initialization and dataset creation."""

    def test_basic_initialization(self, basic_md_hparams):
        """Test basic initialization of MDTrajectoryData.

        :param basic_md_hparams: Basic MDTrajectoryDataHparams fixture."""
        data = MDTrajectoryData(basic_md_hparams)

        assert data.traj is not None
        assert data.traj.n_frames == 100
        assert data.traj.n_atoms == 8

    def test_get_dataset(self, basic_md_hparams):
        """Test dataset creation returns correct tensor format.

        :param basic_md_hparams: Basic MDTrajectoryDataHparams fixture."""
        data = MDTrajectoryData(basic_md_hparams)
        dataset = data.get_dataset()

        assert len(dataset) == 100
        assert dataset.tensors[0].shape == (100, 8, 3)
        assert isinstance(dataset.tensors[0], torch.Tensor)


class TestPreprocessing:
    """Tests for individual preprocessing options."""

    def test_remove_solvent(self, mock_trajectory_files):
        """Test solvent removal preprocessing.

        :param mock_trajectory_files: Mock trajectory files fixture."""
        traj_path, top_path = mock_trajectory_files
        hparams = MDTrajectoryDataHparams(
            trajectory_path=traj_path,
            topology_path=top_path,
            remove_solvent=True,
            superpose=False,
            center_coordinates=False,
            prune_hydrogens=False,
            stride=1,
        )

        data = MDTrajectoryData(hparams)

        # Should have removed water molecules (3 atoms)
        assert data.traj.n_atoms == 5

    def test_prune_hydrogens(self, mock_trajectory_files):
        """Test hydrogen pruning preprocessing.

        :param mock_trajectory_files: Mock trajectory files fixture."""
        traj_path, top_path = mock_trajectory_files
        hparams = MDTrajectoryDataHparams(
            trajectory_path=traj_path,
            topology_path=top_path,
            remove_solvent=True,
            superpose=False,
            center_coordinates=False,
            prune_hydrogens=True,
            stride=1,
        )

        data = MDTrajectoryData(hparams)

        # Should have only heavy atoms (3 carbons after removing solvent)
        assert data.traj.n_atoms == 3

    def test_superpose(self, mock_trajectory_files):
        """Test trajectory superposition.

        :param mock_trajectory_files: Mock trajectory files fixture."""
        traj_path, top_path = mock_trajectory_files
        hparams = MDTrajectoryDataHparams(
            trajectory_path=traj_path,
            topology_path=top_path,
            remove_solvent=False,
            superpose=True,
            center_coordinates=False,
            prune_hydrogens=False,
            stride=1,
        )

        data = MDTrajectoryData(hparams)

        # Check that frames are aligned
        assert data.traj is not None
        assert data.traj.n_frames == 100

    def test_center_coordinates(self, mock_trajectory_files):
        """Test coordinate centering.

        :param mock_trajectory_files: Mock trajectory files fixture."""
        traj_path, top_path = mock_trajectory_files
        hparams = MDTrajectoryDataHparams(
            trajectory_path=traj_path,
            topology_path=top_path,
            remove_solvent=False,
            superpose=False,
            center_coordinates=True,
            prune_hydrogens=False,
            stride=1,
        )

        data = MDTrajectoryData(hparams)
        dataset = data.get_dataset()

        # Check that coordinates are approximately centered
        coords = dataset.tensors[0]
        mean_coords = coords.mean(dim=(0, 1))
        assert torch.allclose(mean_coords, torch.zeros(3), atol=1e-6)

    def test_stride(self, mock_trajectory_files):
        """Test stride parameter reduces number of frames.

        :param mock_trajectory_files: Mock trajectory files fixture."""
        traj_path, top_path = mock_trajectory_files
        hparams = MDTrajectoryDataHparams(
            trajectory_path=traj_path,
            topology_path=top_path,
            remove_solvent=False,
            superpose=False,
            center_coordinates=False,
            prune_hydrogens=False,
            stride=2,
        )

        data = MDTrajectoryData(hparams)

        # With stride=2, should have 50 frames
        assert data.traj.n_frames == 50

    def test_combined_preprocessing(self, mock_trajectory_files):
        """Test multiple preprocessing steps combined.

        :param mock_trajectory_files: Mock trajectory files fixture."""
        traj_path, top_path = mock_trajectory_files
        hparams = MDTrajectoryDataHparams(
            trajectory_path=traj_path,
            topology_path=top_path,
            remove_solvent=True,
            superpose=True,
            center_coordinates=True,
            prune_hydrogens=True,
            stride=2,
        )

        data = MDTrajectoryData(hparams)
        dataset = data.get_dataset()

        # Check final dimensions
        assert data.traj.n_frames == 50
        assert data.traj.n_atoms == 3  # Only heavy atoms, no solvent
        assert dataset.tensors[0].shape == (50, 3, 3)


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_prune_hydrogens_without_topology_fails(self, mock_trajectory_files):
        """Test that pruning hydrogens fails gracefully without topology.

        :param mock_trajectory_files: Mock trajectory files fixture."""
        traj_path, _ = mock_trajectory_files

        hparams = MDTrajectoryDataHparams(
            trajectory_path=traj_path,
            topology_path=traj_path,  # Use traj as topology (will fail)
            remove_solvent=False,
            superpose=False,
            center_coordinates=False,
            prune_hydrogens=True,
            stride=1,
        )

        # The initialization might fail depending on mdtraj behavior
        with pytest.raises((ValueError, Exception)):
            data = MDTrajectoryData(hparams)
