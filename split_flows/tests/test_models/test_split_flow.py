import torch


class TestNoiseAugmentation:
    """Tests for NoiseAugmentation class."""

    def test_init(self, latent_groupings):
        """Test NoiseAugmentation initialization.

        :param latent_groupings: Latent groupings fixture.
        :return: None."""
        from split_flows.models.split_flow import NoiseAugmentation

        noise = NoiseAugmentation(
            latent_groupings=latent_groupings,
            scale=0.1,
            num_particles=9,
        )
        assert noise.scale == 0.1
        assert noise.num_particles == 9
        assert len(noise.latent_groupings) == 3

    def test_augment_shape(self, noise_augmentation, sample_cg_coords):
        """Test that augment produces correct output shape.

        :param noise_augmentation: NoiseAugmentation fixture.
        :param sample_cg_coords: Sample CG coordinates fixture.
        :return: None."""
        result = noise_augmentation.augment(sample_cg_coords)
        assert result.shape == (4, 9, 3)

    def test_augment_preserves_cg_coords(self, noise_augmentation, sample_cg_coords):
        """Test that CG coordinates are preserved in augmentation.

        :param noise_augmentation: NoiseAugmentation fixture.
        :param sample_cg_coords: Sample CG coordinates fixture.
        :return: None."""
        result = noise_augmentation.augment(sample_cg_coords)
        assert torch.allclose(result[:, 0, :], sample_cg_coords[:, 0, :])
        assert torch.allclose(result[:, 3, :], sample_cg_coords[:, 1, :])
        assert torch.allclose(result[:, 6, :], sample_cg_coords[:, 2, :])

    def test_augment_temperature_scaling(self, noise_augmentation, sample_cg_coords):
        """Test that temperature parameter scales noise correctly.

        :param noise_augmentation: NoiseAugmentation fixture.
        :param sample_cg_coords: Sample CG coordinates fixture.
        :return: None."""
        torch.manual_seed(42)
        result_t1 = noise_augmentation.augment(sample_cg_coords, temperature=1.0)
        torch.manual_seed(42)
        result_t2 = noise_augmentation.augment(sample_cg_coords, temperature=2.0)

        noise_diff_t1 = result_t1[:, 1, :] - sample_cg_coords[:, 0, :]
        noise_diff_t2 = result_t2[:, 1, :] - sample_cg_coords[:, 0, :]

        assert torch.mean(torch.abs(noise_diff_t2)) > torch.mean(torch.abs(noise_diff_t1))

    def test_to_standard_normal_shape(self, noise_augmentation, sample_aa_coords):
        """Test to_standard_normal output shape.

        :param noise_augmentation: NoiseAugmentation fixture.
        :param sample_aa_coords: Sample AA coordinates fixture.
        :return: None."""
        result = noise_augmentation.to_standard_normal(sample_aa_coords)
        assert result.shape == sample_aa_coords.shape

    def test_to_standard_normal_preserves_cg(self, noise_augmentation, sample_aa_coords):
        """Test that to_standard_normal preserves CG coordinates.

        :param noise_augmentation: NoiseAugmentation fixture.
        :param sample_aa_coords: Sample AA coordinates fixture.
        :return: None."""
        result = noise_augmentation.to_standard_normal(sample_aa_coords)
        assert torch.allclose(result[:, 0, :], sample_aa_coords[:, 0, :])
        assert torch.allclose(result[:, 3, :], sample_aa_coords[:, 3, :])
        assert torch.allclose(result[:, 6, :], sample_aa_coords[:, 6, :])

    def test_log_prob_shape(self, noise_augmentation, sample_aa_coords):
        """Test log_prob output shape.

        :param noise_augmentation: NoiseAugmentation fixture.
        :param sample_aa_coords: Sample AA coordinates fixture.
        :return: None."""
        log_prob = noise_augmentation.log_prob(sample_aa_coords)
        assert log_prob.shape == (sample_aa_coords.shape[0],)

    def test_log_prob_values(self, noise_augmentation, sample_cg_coords):
        """Test that log_prob returns reasonable values.

        :param noise_augmentation: NoiseAugmentation fixture.
        :param sample_cg_coords: Sample CG coordinates fixture.
        :return: None."""
        augmented = noise_augmentation.augment(sample_cg_coords)
        log_prob = noise_augmentation.log_prob(augmented)

        assert torch.all(log_prob < 0)
        assert torch.all(torch.isfinite(log_prob))


class TestVelocityNet:
    """Tests for VelocityNet class."""

    def test_init(self):
        """Test VelocityNet initialization.

        :return: None."""
        from split_flows.models.split_flow import VelocityNet

        atom_types = torch.randn(9, 5)
        bead_types = torch.randn(9, 3)
        net = VelocityNet(
            dim=16,
            depth=2,
            fourier_features=8,
            atom_types=atom_types,
            bead_types=bead_types,
        )
        assert net.dim == 16
        assert net.depth == 2
        assert net.fourier_features == 8

    def test_forward_shape(self, velocity_net, sample_aa_coords):
        """Test that forward pass produces correct output shape.

        :param velocity_net: VelocityNet fixture.
        :param sample_aa_coords: Sample AA coordinates fixture.
        :return: None."""
        t = torch.rand(sample_aa_coords.shape[0], 1, 1)
        output = velocity_net(sample_aa_coords, t)
        assert output.shape == sample_aa_coords.shape

    def test_forward_batch_consistency(self, velocity_net):
        """Test that velocity is consistent across batch dimension.

        :param velocity_net: VelocityNet fixture.
        :return: None."""
        x = torch.randn(1, 9, 3)
        t = torch.tensor([[[0.5]]])

        output_single = velocity_net(x, t)

        x_batch = x.repeat(4, 1, 1)
        t_batch = t.repeat(4, 1, 1)
        output_batch = velocity_net(x_batch, t_batch)

        assert torch.allclose(output_single[0], output_batch[0], atol=1e-5)

    def test_forward_time_dependence(self, trained_velocity_net, sample_aa_coords):
        """Test that velocity changes with time.

        :param velocity_net: VelocityNet fixture.
        :param sample_aa_coords: Sample AA coordinates fixture.
        :return: None."""
        t1 = torch.tensor([[[0.1]]])
        t2 = torch.tensor([[[0.9]]])

        x = sample_aa_coords[:1]
        output_t1 = trained_velocity_net(x, t1)
        output_t2 = trained_velocity_net(x, t2)

        assert not torch.allclose(output_t1, output_t2)


class TestSplitFlow:
    """Tests for SplitFlow class."""

    def test_split_flow_init(self, split_flow):
        """Test SplitFlow initialization.

        :param split_flow: SplitFlow fixture.
        :return: None."""
        assert split_flow.num_particles == 9
        assert split_flow.noise.scale == 0.1

    def test_num_particles_property(self, split_flow):
        """Test num_particles property calculation.

        :param split_flow: SplitFlow fixture.
        :return: None."""
        assert split_flow.num_particles == 9

    def test_indices_split_property(self, split_flow):
        """Test indices_split property.

        :param split_flow: SplitFlow fixture.
        :return: None."""
        cg_indices, noise_indices = split_flow.indices_split

        assert cg_indices.shape == (3,)
        assert noise_indices.shape == (6,)
        assert torch.allclose(cg_indices, torch.tensor([0, 3, 6]))
        assert torch.allclose(noise_indices, torch.tensor([1, 2, 4, 5, 7, 8]))

    def test_velocity_shape(self, split_flow):
        """Test velocity method output shape.

        :param split_flow: SplitFlow fixture.
        :return: None."""
        xt = torch.randn(4, 9, 3)
        t = torch.rand(4, 1, 1)

        vt = split_flow.velocity(xt, t)
        assert vt.shape == xt.shape

    def test_velocity_time_broadcasting(self, split_flow):
        """Test that velocity handles time broadcasting correctly.

        :param split_flow: SplitFlow fixture.
        :return: None."""
        xt = torch.randn(4, 9, 3)
        t_single = torch.rand(1, 1, 1)

        vt = split_flow.velocity(xt, t_single)
        assert vt.shape == xt.shape

    def test_compute_metrics_structure(self, split_flow):
        """Test compute_metrics returns correct structure.

        :param split_flow: SplitFlow fixture.
        :return: None."""
        split_flow.eval()

        batch = (torch.randn(4, 9, 3),)
        metrics = split_flow.compute_metrics(batch, batch_idx=0)

        assert "loss" in metrics
        assert "loss_fm" in metrics
        assert metrics["loss"].ndim == 0
        assert metrics["loss_fm"].ndim == 0

    def test_compute_metrics_loss_positive(self, split_flow):
        """Test that computed loss is positive.

        :param split_flow: SplitFlow fixture.
        :return: None."""
        split_flow.eval()

        batch = (torch.randn(4, 9, 3),)
        metrics = split_flow.compute_metrics(batch, batch_idx=0)

        assert metrics["loss"] >= 0
        assert metrics["loss_fm"] >= 0
