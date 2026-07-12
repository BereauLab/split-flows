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
    output: Tensor,     #velocity field
    x: Tensor,          #the position of system at time t
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
    grad = torch.autograd.grad(
        output, x, grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad


def hutchinson_trace(
    output: Tensor,
    x: Tensor,
    num_samples: int = 100,
) -> Tensor:
    """Compute the trace of the Jacobian using Hutchinson trace estimator.
    
    Estimates tr(J) = E[z^T * J * z] where z ~ N(0, I).
    This is much more efficient than computing the full Jacobian and avoids
    graph retention issues with autograd.grad() in loops.  
    
    :param output: Output tensor (typically velocity field), shape (batch, dim)
    :param x: Input tensor with respect to which Jacobian is computed, shape (batch, dim)
    :param num_samples: Number of random samples for trace estimation (default: 100)
    :return: Estimated trace (divergence) for each sample in batch, shape (batch,)
    """
    batch_size, dim = output.shape
    trace_estimate = torch.zeros(batch_size, device=output.device, dtype=output.dtype)
    
    for i in range(num_samples):
        # Sample random vector z ~ N(0, I)
        z = torch.randn_like(output)
        
        #only retain graph if the output is needed for the next iteration:
        is_last_sample = (i == num_samples - 1)

        # Compute z^T * J by computing gradients of (output * z).sum()
        # This gives us one row of the Jacobian contracted with z
        jvp = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=z,
            create_graph=False,
            retain_graph=not is_last_sample,
        )[0]
        
        # Compute z^T * (J * z) = (z * jvp).sum(dim=-1)
        # This estimates one sample of tr(J)
        trace_estimate += (z * jvp).sum(dim=-1)
    
    return trace_estimate / num_samples
