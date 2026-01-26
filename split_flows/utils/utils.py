import numpy as np
from numpy import ndarray

import torch
from torch import Tensor


def expand_like(input: Tensor, reference: Tensor) -> Tensor:
    """Expands the input tensor to match the shape of the reference tensor.

    :param input: tensor to be expanded.
    :param reference: tensor to match the shape."""

    expanded = input.clone()
    while expanded.dim() < reference.dim():
        expanded = expanded.unsqueeze(-1)
    return expanded.expand_as(reference)


def match_dims(input: Tensor, reference: Tensor) -> Tensor:
    """Matches the dimensions of the input tensor to the reference tensor by unsqueezing
    the input tensor until it has the same number of dimensions as the reference tensor.

    :param input: tensor to be matched.
    :param reference: tensor to match the dimensions."""

    while input.dim() < reference.dim():
        input = input.unsqueeze(-1)
    return input


def sum_except_batch(tensor: Tensor) -> Tensor:
    """Sums all dimensions of a tensor except the batch dimension.

    :param tensor: tensor to sum.
    :return: tensor with the sum of all dimensions except the batch dimension."""

    return tensor.view(tensor.size(0), -1).sum(dim=1)


def to_one_hot(
    input_list: Tensor | ndarray | list,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if isinstance(input_list, Tensor):
        input_list = input_list.numpy()
    elif isinstance(input_list, list):
        input_list = np.array(input_list)
    elif isinstance(input_list, ndarray):
        pass
    else:
        raise ValueError("The input must be of type Tensor or ndarray or list.")

    unique_elements = np.unique(input_list)
    one_hot_basis = np.eye(len(unique_elements))
    return torch.tensor(
        [
            one_hot_basis[np.where(unique_elements == element)[0][0]].tolist()
            for element in input_list
        ],
        dtype=dtype,
        device=device,
    )


def gradient(
    output: Tensor,
    x: Tensor,
    grad_outputs: Tensor | None = None,
    create_graph: bool = False,
) -> Tensor:
    """Computes the gradient of the output with respect to the input x.

    :param output: the output tensor.
    :param x: the input tensor with respect to which the gradient is computed.
    :param grad_outputs: optional tensor to specify the gradient of the output.
    :param create_graph: whether to create the graph for the gradient computation.
    :return: the gradient of the output with respect to the input x."""

    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(output, x, grad_outputs=grad_outputs, create_graph=create_graph)[0]
    return grad
