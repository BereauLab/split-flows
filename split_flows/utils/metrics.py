from tqdm import tqdm

import mdtraj as md
import numpy as np
import torch


def compute_cg_rmsd(
    traj1: md.Trajectory, traj2: md.Trajectory, indices: list[int], splits: int = 5
) -> list[float]:
    """Computes the root mean squared deviation of configurations of the provided trajectories
    in the coarse-grained space.

    :param traj1: first trajectory
    :param traj2: second trajectory
    :param indices: indices of atoms to retain in the coarse-grained representation
    :param splits: number of splits to average over
    :return: list of RMSDs of coarse-grained representations per split"""

    split_size = traj1.n_frames // splits

    rmsd_list = []
    for i in range(splits):
        rmsd_list_split = []
        start = i * split_size
        end = (i + 1) * split_size if i < splits - 1 else traj1.n_frames
        for j in tqdm(range(start, end)):
            rmsd_list_split.append(md.rmsd(traj2[j], traj1[j], 0, atom_indices=indices))

        rmsd_list.append(np.mean(rmsd_list_split))

    return rmsd_list


COVCUTOFFTABLE = {
    1: 0.23,
    2: 0.93,
    3: 0.68,
    4: 0.35,
    5: 0.83,
    6: 0.68,
    7: 0.68,
    8: 0.68,
    9: 0.64,
    10: 1.12,
    11: 0.97,
    12: 1.1,
    13: 1.35,
    14: 1.2,
    15: 0.75,
    16: 1.02,
    17: 0.99,
    18: 1.57,
    19: 1.33,
    20: 0.99,
    21: 1.44,
    22: 1.47,
    23: 1.33,
    24: 1.35,
    25: 1.35,
    26: 1.34,
    27: 1.33,
    28: 1.5,
    29: 1.52,
    30: 1.45,
    31: 1.22,
    32: 1.17,
    33: 1.21,
    34: 1.22,
    35: 1.21,
    36: 1.91,
    37: 1.47,
    38: 1.12,
    39: 1.78,
    40: 1.56,
    41: 1.48,
    42: 1.47,
    43: 1.35,
    44: 1.4,
    45: 1.45,
    46: 1.5,
    47: 1.59,
    48: 1.69,
    49: 1.63,
    50: 1.46,
    51: 1.46,
    52: 1.47,
    53: 1.4,
    54: 1.98,
    55: 1.67,
    56: 1.34,
    57: 1.87,
    58: 1.83,
    59: 1.82,
    60: 1.81,
    61: 1.8,
    62: 1.8,
    63: 1.99,
    64: 1.79,
    65: 1.76,
    66: 1.75,
    67: 1.74,
    68: 1.73,
    69: 1.72,
    70: 1.94,
    71: 1.72,
    72: 1.57,
    73: 1.43,
    74: 1.37,
    75: 1.35,
    76: 1.37,
    77: 1.32,
    78: 1.5,
    79: 1.5,
    80: 1.7,
    81: 1.55,
    82: 1.54,
    83: 1.54,
    84: 1.68,
    85: 1.7,
    86: 2.4,
    87: 2.0,
    88: 1.9,
    89: 1.88,
    90: 1.79,
    91: 1.61,
    92: 1.58,
    93: 1.55,
    94: 1.53,
    95: 1.51,
    96: 1.5,
    97: 1.5,
    98: 1.5,
    99: 1.5,
    100: 1.5,
    101: 1.5,
    102: 1.5,
    103: 1.5,
    104: 1.57,
    105: 1.49,
    106: 1.43,
    107: 1.41,
}


def compute_bond_cutoff_mdtraj(topology, scale=1.3):
    """Compute bond cutoffs for MDTraj topology"""
    atomic_nums = [atom.element.atomic_number for atom in topology.atoms]
    # COVCUTOFFTABLE values are in Angstroms, convert to nanometers for MDTraj
    vdw_array = torch.Tensor(
        [COVCUTOFFTABLE[int(el)] / 10.0 for el in atomic_nums]
    )  # Å to nm

    cutoff_array = (vdw_array[None, :] + vdw_array[:, None]) * scale

    return cutoff_array


def compute_distance_mat_mdtraj(xyz, device="cpu"):
    """Compute distance matrix from xyz coordinates"""
    xyz_tensor = torch.Tensor(xyz).to(device)
    dist = (xyz_tensor[:, None, :] - xyz_tensor[None, :, :]).pow(2).sum(-1).sqrt()

    return dist


def get_bond_graphs_mdtraj(traj, frame_idx=0, device="cpu", scale=1.3):
    """Get bond graph for a specific frame in MDTraj trajectory"""
    xyz = traj.xyz[frame_idx]  # coordinates for specific frame
    dist = compute_distance_mat_mdtraj(xyz, device=device)
    cutoff = compute_bond_cutoff_mdtraj(traj.topology, scale=scale)
    bond_mat = dist < cutoff.to(device)
    bond_mat[np.diag_indices(traj.n_atoms)] = 0

    del dist, cutoff

    return bond_mat.to(torch.long).to("cpu")


def compare_graph_mdtraj(ref_traj, traj, ref_frame=0, frame=0, scale=1.3):
    """Compare bond graphs between two MDTraj trajectory frames"""
    ref_bonds = get_bond_graphs_mdtraj(ref_traj, frame_idx=ref_frame, scale=scale)
    bonds = get_bond_graphs_mdtraj(traj, frame_idx=frame, scale=scale)

    diff = (bonds != ref_bonds).sum().item()

    return diff


def graph_edit_distance(
    traj: md.Trajectory, scale: float = 1.3, verbose: bool = True
) -> list[float]:
    """Compare trajectory bond graphs to topology bonds."""
    n_atoms = traj.n_atoms
    # Create reference adjacency matrix from topology bonds
    A_ref = np.zeros((n_atoms, n_atoms), dtype=int)
    for bond in traj.topology.bonds:
        i, j = bond[0].index, bond[1].index
        A_ref[i, j] = 1
        A_ref[j, i] = 1
    n_bonds = A_ref.sum()

    ged_list = []
    iterator = range(traj.n_frames)
    if verbose:
        iterator = tqdm(iterator, desc="Computing GED")
    for i in iterator:
        A = get_bond_graphs_mdtraj(traj, frame_idx=i, scale=scale).numpy()
        ged = np.abs((A - A_ref).sum()) / n_bonds
        ged_list.append(ged)
    return ged_list
