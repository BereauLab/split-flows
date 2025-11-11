import torch

import mdtraj
from mdtraj import Trajectory

from torch.utils.data import TensorDataset

from hydrantic.data import PyTorchData
from hydrantic.data.hparams import DataHparams


class MDTrajectoryDataHparams(DataHparams):
    trajectory_path: str
    topology_path: str

    remove_solvent: bool = True
    superpose: bool = True
    center_coordinates: bool = True
    prune_hydrogens: bool = False
    stride: int = 1


class MDTrajectoryData(PyTorchData):
    hparams_schema = MDTrajectoryDataHparams
    traj: Trajectory

    def __init__(self, thparams: MDTrajectoryDataHparams):
        self.thparams = thparams
        super(MDTrajectoryData, self).__init__(thparams)

    def pre_init(self):
        self.traj = mdtraj.load(
            self.thparams.trajectory_path,
            top=self.thparams.topology_path,
            stride=self.thparams.stride,
        )

        if self.thparams.remove_solvent:
            self.traj = self.traj.remove_solvent(inplace=False)

        if self.thparams.superpose:
            self.traj = self.traj.superpose(self.traj[0])

        if self.thparams.center_coordinates:
            self.traj = self.traj.center_coordinates()

        if self.thparams.prune_hydrogens:
            if self.traj.topology is None:
                raise ValueError("Cannot prune hydrogens if topology is not available.")
            heavy_atom_indices = self.traj.topology.select("not element H")
            self.traj = self.traj.atom_slice(heavy_atom_indices)

    def get_dataset(self) -> TensorDataset:
        traj_xyz = torch.tensor(self.traj.xyz)
        return TensorDataset(traj_xyz)
