import math

import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):
    """
    Stores the coefficients of the forward diffusion process.

    The implementation supports the two noise schedules currently considered
    in the project:
        - "linear"
        - "cosine"


    """

    def __init__(
        self,
        num_timesteps,
        schedule_type,
        beta_start=1e-4,
        beta_end=2e-2,
        cosine_s=0.008,
    ):
        super().__init__()

        if not isinstance(num_timesteps, int) or num_timesteps <= 0:
            raise ValueError("num_timesteps must be a positive integer.")

        if schedule_type not in ("linear", "cosine"):
            raise ValueError(
                "schedule_type must be explicitly set to 'linear' or 'cosine'."
            )

        if not (0.0 < beta_start < 1.0):
            raise ValueError("beta_start must be in (0, 1).")

        if not (0.0 < beta_end < 1.0):
            raise ValueError("beta_end must be in (0, 1).")

        if schedule_type == "linear" and beta_start >= beta_end:
            raise ValueError(
                "For a linear schedule, beta_start must be smaller than beta_end."
            )

        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        betas = self._build_betas(
            num_timesteps=num_timesteps,
            schedule_type=schedule_type,
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_s=cosine_s,
        )

        # alpha_t = 1 - beta_t
        alphas = 1.0 - betas

        # alpha_bar_t = product_{s=0}^t alpha_s
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        # Terms used directly by q(x_t | x_0):
        #
        # x_t = sqrt(alpha_bar_t) * x_0
        #       + sqrt(1 - alpha_bar_t) * epsilon
        self.register_buffer(
            "sqrt_alpha_bars",
            torch.sqrt(alpha_bars),
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )

    @staticmethod
    def _build_betas(
        num_timesteps,
        schedule_type,
        beta_start,
        beta_end,
        cosine_s,
    ):
        if schedule_type == "linear":
            return torch.linspace(
                beta_start,
                beta_end,
                num_timesteps,
                dtype=torch.float32,
            )

        # Improved-DDPM style cosine schedule.
        # alpha_bar(t) describes the cumulative signal remaining at time t.
        steps = torch.arange(
            num_timesteps + 1,
            dtype=torch.float64,
        )

        time = steps / num_timesteps

        alpha_bar = torch.cos(
            ((time + cosine_s) / (1.0 + cosine_s))
            * math.pi
            / 2.0
        ) ** 2

        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]

        # Prevent numerical/pathological values at the final step.
        betas = betas.clamp(min=1e-5, max=0.999)

        return betas.to(dtype=torch.float32)

    def _extract(self, values, timesteps, reference):
        """
        Extract one scalar coefficient per batch element and reshape it so
        that it broadcasts over an image tensor.

        values:
            [T]
        timesteps:
            [B]
        reference:
            [B, C, H, W] (or another tensor with batch dimension first)

        returns:
            [B, 1, 1, 1] for image tensors
        """
        if timesteps.dim() != 1:
            raise ValueError("timesteps must have shape [B].")

        if timesteps.size(0) != reference.size(0):
            raise ValueError(
                "timesteps batch size must match the reference batch size."
            )

        coefficients = values[timesteps]

        shape = (timesteps.size(0),) + (1,) * (reference.dim() - 1)

        return coefficients.reshape(shape)

    def sqrt_alpha_bar(self, timesteps, reference):
        """Return sqrt(alpha_bar_t) broadcastable to reference."""
        return self._extract(
            self.sqrt_alpha_bars,
            timesteps,
            reference,
        )

    def sqrt_one_minus_alpha_bar(self, timesteps, reference):
        """Return sqrt(1 - alpha_bar_t) broadcastable to reference."""
        return self._extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            reference,
        )

    def sample_timesteps(self, batch_size, device=None):
        """
        Sample independent timesteps uniformly from {0, ..., T-1}.

        Returns:
            Tensor of shape [B], dtype torch.long.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        return torch.randint(
            low=0,
            high=self.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def reverse_step(self, x_t, predicted_noise, timesteps):
        """
        Performs one reverse DDPM step.

        Given x_t and the noise predicted by the U-Net,
        computes x_{t-1}.

        Args:
            x_t:
                Noisy image at timestep t.
                Shape: [B, C, H, W]

            predicted_noise:
                Noise predicted by the model.
                Shape: [B, C, H, W]

            timesteps:
                Current timestep for each batch element.
                Shape: [B]

        Returns:
            x_{t-1}: [B, C, H, W]
        """
        if x_t.dim() != 4:
            raise ValueError("x_t must have shape [B, C, H, W].")

        if predicted_noise.shape != x_t.shape:
            raise ValueError(
                "predicted_noise must have the same shape as x_t."
            )

        if timesteps.dtype != torch.long:
            raise ValueError("timesteps must have dtype torch.long.")

        if timesteps.dim() != 1:
            raise ValueError("timesteps must have shape [B].")

        if timesteps.size(0) != x_t.size(0):
            raise ValueError(
                "timesteps batch size must match x_t batch size."
            )

        if torch.any(timesteps < 0) or torch.any(
            timesteps >= self.num_timesteps
        ):
            raise ValueError("timesteps contain an invalid value.")

        beta_t = self._extract(
            self.betas,
            timesteps,
            reference=x_t,
        )

        alpha_t = self._extract(
            self.alphas,
            timesteps,
            reference=x_t,
        )

        alpha_bar_t = self._extract(
            self.alpha_bars,
            timesteps,
            reference=x_t,
        )

        # DDPM mean:
        # 1 / sqrt(alpha_t) *
        # (
        #     x_t -
        #     beta_t / sqrt(1 - alpha_bar_t) * predicted_noise
        # )
        mean = (
            1.0 / torch.sqrt(alpha_t)
        ) * (
            x_t
            - (
                beta_t
                / torch.sqrt(1.0 - alpha_bar_t)
            ) * predicted_noise
        )

        # For t = 0 there is no previous stochastic step.
        # We directly return the mean.
        if torch.all(timesteps == 0):
            return mean

        # Posterior variance:
        # beta_tilde_t =
        # beta_t * (1 - alpha_bar_{t-1})
        # / (1 - alpha_bar_t)
        alpha_bar_previous = torch.ones_like(alpha_bar_t)

        nonzero_mask = timesteps > 0

        previous_timesteps = (timesteps - 1).clamp(min=0)

        alpha_bar_previous[nonzero_mask] = self._extract(
            self.alpha_bars,
            previous_timesteps[nonzero_mask],
            reference=x_t[nonzero_mask],
        )

        posterior_variance = (
            beta_t
            * (1.0 - alpha_bar_previous)
            / (1.0 - alpha_bar_t)
        )

        noise = torch.randn_like(x_t)

        # No additional noise for t = 0.
        noise_mask = nonzero_mask.float().reshape(
            x_t.size(0),
            *([1] * (x_t.dim() - 1))
        )

        return (
            mean
            + noise_mask
            * torch.sqrt(torch.clamp(posterior_variance, min=0.0))
            * noise
        )


class DiffusionForwardProcess:
    """
    Implements the forward DDPM noising process.

    Given a clean image x_0 and timestep t, samples x_t according to:

        q(x_t | x_0)
          = N(
                sqrt(alpha_bar_t) * x_0,
                (1 - alpha_bar_t) I
            )

    Equivalently:

        x_t =
            sqrt(alpha_bar_t) * x_0
            + sqrt(1 - alpha_bar_t) * epsilon

    where epsilon ~ N(0, I).

    Input:
        x0: [B, C, H, W]
        timesteps: [B]

    Output:
        x_t: [B, C, H, W]
        noise: [B, C, H, W]
    """

    def __init__(self, scheduler):
        if not isinstance(scheduler, DiffusionScheduler):
            raise TypeError(
                "scheduler must be an instance of DiffusionScheduler."
            )

        self.scheduler = scheduler

    def add_noise(self, x0, timesteps, noise=None):
        if x0.dim() != 4:
            raise ValueError("x0 must have shape [B, C, H, W].")

        if timesteps.dtype != torch.long:
            raise ValueError("timesteps must have dtype torch.long.")

        if timesteps.dim() != 1:
            raise ValueError("timesteps must have shape [B].")

        if timesteps.size(0) != x0.size(0):
            raise ValueError(
                "timesteps batch size must match x0 batch size."
            )

        if torch.any(timesteps < 0) or torch.any(
            timesteps >= self.scheduler.num_timesteps
        ):
            raise ValueError("timesteps contain an invalid value.")

        if noise is None:
            noise = torch.randn_like(x0)
        elif noise.shape != x0.shape:
            raise ValueError("noise must have the same shape as x0.")

        sqrt_alpha_bar = self.scheduler.sqrt_alpha_bar(
            timesteps,
            reference=x0,
        )

        sqrt_one_minus_alpha_bar = (
            self.scheduler.sqrt_one_minus_alpha_bar(
                timesteps,
                reference=x0,
            )
        )

        xt = (
            sqrt_alpha_bar * x0
            + sqrt_one_minus_alpha_bar * noise
        )

        return xt, noise

    def sample(self, x0, timesteps):
        """
        Convenience wrapper for training code.

        Returns:
            xt, noise
        """
        return self.add_noise(x0, timesteps)
