import pytest
import numpy as np
import torch

from split_flows.utils.utils import (
    expand_like,
    match_dims,
    sum_except_batch,
    to_one_hot,
    gradient,
)


class TestExpandLike:
    """Tests for expand_like function."""

    def test_expand_scalar_to_2d(self):
        """Test expanding a scalar tensor to match a 2D tensor.

        :return: None."""
        input_tensor = torch.tensor(5.0)
        reference = torch.ones(3, 4)
        result = expand_like(input_tensor, reference)
        assert result.shape == reference.shape
        assert torch.all(result == 5.0)

    def test_expand_1d_to_3d(self):
        """Test expanding a 1D tensor to match a 3D tensor.

        :return: None."""
        input_tensor = torch.tensor([1.0, 2.0])
        reference = torch.ones(2, 3, 4)
        result = expand_like(input_tensor, reference)
        assert result.shape == reference.shape

    def test_expand_same_dims(self):
        """Test that tensors with same dimensions are unchanged.

        :return: None."""
        input_tensor = torch.ones(2, 3)
        reference = torch.zeros(2, 3)
        result = expand_like(input_tensor, reference)
        assert result.shape == input_tensor.shape
        assert torch.all(result == 1.0)


class TestMatchDims:
    """Tests for match_dims function."""

    def test_match_scalar_to_2d(self):
        """Test matching dimensions of scalar to 2D tensor.

        :return: None."""
        input_tensor = torch.tensor(5.0)
        reference = torch.ones(3, 4)
        result = match_dims(input_tensor, reference)
        assert result.dim() == reference.dim()
        assert result.shape == (1, 1)

    def test_match_1d_to_3d(self):
        """Test matching dimensions of 1D to 3D tensor.

        :return: None."""
        input_tensor = torch.tensor([1.0, 2.0])
        reference = torch.ones(2, 3, 4)
        result = match_dims(input_tensor, reference)
        assert result.dim() == reference.dim()
        assert result.shape == (2, 1, 1)

    def test_match_same_dims(self):
        """Test that tensors with same number of dimensions are unchanged.

        :return: None."""
        input_tensor = torch.ones(2, 3)
        reference = torch.zeros(5, 7)
        result = match_dims(input_tensor, reference)
        assert result.shape == input_tensor.shape


class TestSumExceptBatch:
    """Tests for sum_except_batch function."""

    def test_sum_2d_tensor(self):
        """Test summing 2D tensor along all dims except batch.

        :return: None."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = sum_except_batch(tensor)
        expected = torch.tensor([6.0, 15.0])
        assert torch.allclose(result, expected)

    def test_sum_3d_tensor(self):
        """Test summing 3D tensor along all dims except batch.

        :return: None."""
        tensor = torch.ones(2, 3, 4)
        result = sum_except_batch(tensor)
        expected = torch.tensor([12.0, 12.0])
        assert torch.allclose(result, expected)

    def test_sum_single_batch(self):
        """Test summing with single batch element.

        :return: None."""
        tensor = torch.tensor([[1.0, 2.0, 3.0]])
        result = sum_except_batch(tensor)
        assert result.shape == (1,)
        assert result[0] == 6.0


class TestToOneHot:
    """Tests for to_one_hot function."""

    def test_one_hot_from_list(self):
        """Test converting list to one-hot encoding.

        :return: None."""
        input_list = [0, 1, 2, 1, 0]
        result = to_one_hot(input_list)
        assert result.shape == (5, 3)
        assert torch.allclose(result[0], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.allclose(result[1], torch.tensor([0.0, 1.0, 0.0]))
        assert torch.allclose(result[2], torch.tensor([0.0, 0.0, 1.0]))

    def test_one_hot_from_numpy(self):
        """Test converting numpy array to one-hot encoding.

        :return: None."""
        input_array = np.array([2, 1, 0])
        result = to_one_hot(input_array)
        assert result.shape == (3, 3)
        assert torch.allclose(result[0], torch.tensor([0.0, 0.0, 1.0]))

    def test_one_hot_from_tensor(self):
        """Test converting tensor to one-hot encoding.

        :return: None."""
        input_tensor = torch.tensor([1, 2, 1])
        result = to_one_hot(input_tensor)
        assert result.shape == (3, 2)

    def test_one_hot_with_device_dtype(self):
        """Test one-hot encoding with specified device and dtype.

        :return: None."""
        input_list = [0, 1, 0]
        result = to_one_hot(input_list, device=torch.device("cpu"), dtype=torch.float64)
        assert result.dtype == torch.float64
        assert result.device == torch.device("cpu")

    def test_one_hot_invalid_input(self):
        """Test that invalid input raises ValueError.

        :return: None."""
        with pytest.raises(ValueError, match="The input must be of type"):
            to_one_hot("invalid")  # type: ignore


class TestGradient:
    """Tests for gradient function."""

    def test_gradient_simple_function(self):
        """Test gradient computation for simple quadratic function.

        :return: None."""
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = (x**2).sum()
        grad = gradient(y, x)
        expected = torch.tensor([4.0, 6.0])
        assert torch.allclose(grad, expected)

    def test_gradient_with_grad_outputs(self):
        """Test gradient with custom grad_outputs.

        :return: None."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x**2
        custom_grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        grad = gradient(y, x, grad_outputs=custom_grad)
        expected = torch.tensor([[2.0, 0.0], [0.0, 8.0]])
        assert torch.allclose(grad, expected)

    def test_gradient_create_graph(self):
        """Test gradient with create_graph=True for higher-order derivatives.

        :return: None."""
        x = torch.tensor([2.0], requires_grad=True)
        y = x**3
        grad_1 = gradient(y, x, create_graph=True)
        assert grad_1.requires_grad
        grad_2 = gradient(grad_1, x)
        expected = torch.tensor([12.0])
        assert torch.allclose(grad_2, expected)

    def test_gradient_batched(self):
        """Test gradient computation with batch dimension.

        :return: None."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = (x**2).sum()
        grad = gradient(y, x)
        expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        assert torch.allclose(grad, expected)
