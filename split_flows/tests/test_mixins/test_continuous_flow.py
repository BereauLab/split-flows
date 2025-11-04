import torch


class TestVelocity:
    """Tests for velocity computation."""

    def test_velocity_shape(self, simple_flow, sample_batch):
        """Test that velocity output has correct shape."""
        t = torch.tensor(0.5)
        vt = simple_flow.velocity(sample_batch, t)
        assert vt.shape == sample_batch.shape

    def test_velocity_linear(self, simple_flow):
        """Test linear velocity field returns input."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t = torch.tensor(0.5)
        vt = simple_flow.velocity(x, t)
        assert torch.allclose(vt, x)

    def test_velocity_constant(self, constant_flow, sample_batch):
        """Test constant velocity field."""
        t = torch.tensor(0.5)
        vt = constant_flow.velocity(sample_batch, t)
        expected = torch.ones_like(sample_batch)
        assert torch.allclose(vt, expected)


class TestDivergence:
    """Tests for divergence computation."""

    def test_divergence_shape(self, simple_flow, sample_batch):
        """Test that divergence output has correct shape."""
        t = torch.tensor(0.5)
        vt, div_vt = simple_flow.divergence(sample_batch, t)
        assert vt.shape == sample_batch.shape
        assert div_vt.shape == (sample_batch.shape[0],)

    def test_divergence_linear_flow(self, simple_flow):
        """Test divergence of linear flow v(x) = x equals dimension."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t = torch.tensor(0.5)
        _, div_vt = simple_flow.divergence(x, t)
        expected_div = torch.tensor([2.0, 2.0])
        assert torch.allclose(div_vt, expected_div, atol=1e-5)

    def test_divergence_constant_flow(self, constant_flow):
        """Test divergence of constant flow is zero."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t = torch.tensor(0.5)
        _, div_vt = constant_flow.divergence(x, t)
        expected_div = torch.zeros(2)
        assert torch.allclose(div_vt, expected_div, atol=1e-5)


class TestComputeVolumeChange:
    """Tests for volume change computation."""

    def test_volume_change_shape(self, simple_flow, sample_batch):
        """Test that volume change output has correct shape."""
        x1, log_det = simple_flow.compute_volume_change(sample_batch, step_size=0.1)
        assert x1.shape == sample_batch.shape
        assert log_det.shape == (sample_batch.shape[0],)

    def test_volume_change_chunk_size(self, simple_flow, sample_batch):
        """Test volume change computation with chunking."""
        x1_full, log_det_full = simple_flow.compute_volume_change(
            sample_batch, step_size=0.1
        )
        x1_chunk, log_det_chunk = simple_flow.compute_volume_change(
            sample_batch, chunk_size=5, step_size=0.1
        )
        assert torch.allclose(x1_full, x1_chunk, atol=1e-4)
        assert torch.allclose(log_det_full, log_det_chunk, atol=1e-4)

    def test_volume_change_intermediate(self, simple_flow):
        """Test volume change with intermediate steps."""
        x0 = torch.tensor([[1.0, 2.0]])
        sol, volume_change = simple_flow.compute_volume_change(
            x0, step_size=0.5, return_intermediate=True
        )
        assert sol.shape[0] == 3
        assert volume_change.shape[0] == 3
        assert sol.shape[1:] == x0.shape

    def test_volume_change_methods(self, constant_flow, sample_batch):
        """Test different ODE solver methods."""
        x1_euler, _ = constant_flow.compute_volume_change(
            sample_batch, method="euler", step_size=0.1
        )
        x1_rk4, _ = constant_flow.compute_volume_change(
            sample_batch, method="rk4", step_size=0.1
        )
        assert x1_euler.shape == x1_rk4.shape

    def test_volume_change_constant_flow(self, constant_flow):
        """Test volume change for constant velocity flow.

        For constant velocity v(x,t) = c, divergence is zero everywhere,
        so volume change should be zero (log|det| = 0)."""
        x0 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        _, log_det = constant_flow.compute_volume_change(
            x0, step_size=0.01, method="rk4"
        )

        expected_log_det = torch.zeros(2)
        assert torch.allclose(log_det, expected_log_det, atol=1e-4)

    def test_volume_change_linear_flow(self, simple_flow):
        """Test volume change for linear flow.

        For linear flow v(x,t) = x, divergence is constant = dim.
        The volume change integrates as: log|det| = -int_0^1 div(v) dt = -dim * t.
        At t=1, log|det| = -dim."""
        x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        _, log_det = simple_flow.compute_volume_change(x0, step_size=0.01, method="rk4")

        dim = x0.shape[1]
        expected_log_det = torch.tensor([-float(dim), -float(dim)])
        assert torch.allclose(log_det, expected_log_det, atol=1e-3)

    def test_volume_change_linear_flow_intermediate(self, simple_flow):
        """Test intermediate volume change for linear flow.

        At each time step t, log|det| = -dim * t."""
        x0 = torch.tensor([[1.0, 2.0]])
        step_size = 0.25
        _, volume_change = simple_flow.compute_volume_change(
            x0, step_size=step_size, return_intermediate=True, method="rk4"
        )

        dim = x0.shape[1]
        time_grid = torch.linspace(0, 1, int(1 / step_size) + 1)
        expected_volume_change = -dim * time_grid.unsqueeze(1)

        assert torch.allclose(volume_change, expected_volume_change, atol=1e-3)
