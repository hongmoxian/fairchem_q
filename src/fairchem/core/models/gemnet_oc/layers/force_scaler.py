"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import torch


class ForceScaler:
    """
    Scales up the energy and then scales down the forces
    to prevent NaNs and infs in calculations using AMP.
    Inspired by torch.GradScaler("cuda", args...).
    """

    def __init__(
        self,
        init_scale: float = 2.0**8,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_force_iters: int = 50,
        enabled: bool = True,
    ) -> None:
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_force_iters = max_force_iters
        self.enabled = enabled
        self.finite_force_results = 0

    def scale(self, energy):
        return energy * self.scale_factor if self.enabled else energy

    def unscale(self, forces):
        return forces / self.scale_factor if self.enabled else forces

    def calc_forces(self, energy, pos):
        energy_scaled = self.scale(energy)
        forces_scaled = -torch.autograd.grad(
            energy_scaled,
            pos,
            grad_outputs=torch.ones_like(energy_scaled),
            create_graph=True,
        )[0]
        # (nAtoms, 3)
        return self.unscale(forces_scaled)

    def calc_forces_and_update(self, energy, pos):
        if self.enabled:
            found_nans_or_infs = True
            force_iters = 0

            # Re-calculate forces until everything is nice and finite.
            while found_nans_or_infs:
                forces = self.calc_forces(energy, pos)

                found_nans_or_infs = not torch.all(forces.isfinite())
                if found_nans_or_infs:
                    self.finite_force_results = 0

                    # Prevent infinite loop
                    force_iters += 1
                    if force_iters == self.max_force_iters:
                        logging.warning(
                            "Too many non-finite force results in a batch. "
                            "Breaking scaling loop."
                        )
                        break

                    # Delete graph to save memory
                    del forces
                else:
                    self.finite_force_results += 1
                self.update()
        else:
            forces = self.calc_forces(energy, pos)
        return forces

    def update(self) -> None:
        if self.finite_force_results == 0:
            self.scale_factor *= self.backoff_factor

        if self.finite_force_results == self.growth_interval:
            self.scale_factor *= self.growth_factor
            self.finite_force_results = 0

        logging.info(f"finite force step count: {self.finite_force_results}")
        logging.info(f"scaling factor: {self.scale_factor}")



    def compute_hessian_masked(self, forces, data, training=None, max_samples=None):
        """
        Compute either a full Hessian or a sampled subset of rows.

        During training, if ``max_samples`` is set and smaller than the total
        number of degrees of freedom, a random subset of rows is evaluated and
        returned together with the corresponding row indices. During inference,
        all rows are evaluated.
        """
        if training is None:
            training = False

        pos = data.pos
        n_atoms = pos.shape[0]
        device = pos.device
        n_dofs = 3 * n_atoms

        active_indices = torch.arange(n_dofs, device=device, dtype=torch.long)
        if training:
            if max_samples is not None and len(active_indices) > max_samples:
                perm = torch.randperm(len(active_indices), device=device)
                sampled_indices = active_indices[perm[:max_samples]].sort().values
            else:
                sampled_indices = active_indices
        else:
            sampled_indices = active_indices

        if sampled_indices.numel() == 0:
            return (
                torch.zeros(0, n_dofs, device=device, dtype=forces.dtype),
                sampled_indices,
            )

        if not getattr(data, "hessian_mask", True):
            return (
                torch.zeros(
                    sampled_indices.numel(),
                    n_dofs,
                    device=device,
                    dtype=forces.dtype,
                ),
                sampled_indices,
            )

        forces_flat = forces.flatten()
        num_sampled = int(sampled_indices.numel())
        hessian_rows = torch.zeros(
            num_sampled,
            n_dofs,
            device=device,
            dtype=forces.dtype,
        )

        hessian_clamp_min = -100.0
        hessian_clamp_max = 100.0
        max_gradient_norm = 100.0

        for i, idx in enumerate(sampled_indices):
            v = torch.zeros_like(forces_flat)
            v[idx] = 1.0 

            try:
                grad = torch.autograd.grad(
                    outputs=forces_flat,
                    inputs=pos,
                    grad_outputs=v,
                    retain_graph=training or (i < (num_sampled - 1)),
                    create_graph=training,
                )[0]
                grad_flat = grad.flatten()

                grad_norm = torch.norm(grad_flat)
                if grad_norm > max_gradient_norm:
                    grad_flat = grad_flat / grad_norm * max_gradient_norm
                    logging.warning(
                        "Hessian row %s: gradient norm %.2f clipped to %.2f",
                        int(idx.item()),
                        float(grad_norm.item()),
                        max_gradient_norm,
                    )

                grad_flat = torch.clamp(grad_flat, min=hessian_clamp_min, max=hessian_clamp_max)

                if not torch.all(torch.isfinite(grad_flat)):
                    logging.warning(
                        "Hessian row %s: contains non-finite values, setting to zero",
                        int(idx.item()),
                    )
                    grad_flat = torch.zeros_like(grad_flat)

                hessian_rows[i] = grad_flat

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(
                        "OOM at Hessian row %s, setting to zero",
                        int(idx.item()),
                    )
                else:
                    logging.error(
                        "Error computing Hessian row %s: %s",
                        int(idx.item()),
                        str(e),
                    )

            del v
            if "grad" in locals():
                del grad
            if training and (i % 10 == 0) and torch.cuda.is_available():
                torch.cuda.empty_cache()

        hessian_rows = -hessian_rows
        return hessian_rows, sampled_indices
