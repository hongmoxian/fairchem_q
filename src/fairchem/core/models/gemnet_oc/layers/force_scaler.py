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



    def compute_hessian_masked(self, forces, data, training=False, max_samples=None):
            """
            Args:
                forces: 模型输出的力 [N, 3]
                data: 数据对象
                training: 是否开启 create_graph
                max_samples (int): 随机采样的行数。例如设为 32。
                                如果为 None，则计算所有自由原子的 Hessian。
            
            Returns:
                hessian: [3N, 3N] 稀疏矩阵 (未采样的行为 0)
                row_mask: [3N, 1]  掩码向量，指示哪些行是被计算过的 (用于 Loss 计算)
            """
            pos = data.pos
            n_atoms = pos.shape[0]
            device = pos.device
            
            # 1. 确定自由原子 (Fixed=0 为 Free)
            if hasattr(data, 'fixed') and data.fixed is not None:
                # fixed: 1=Fixed, 0=Free -> is_free: 1=Free
                is_free = (data.fixed == 0).float()
                # 扩展 Mask: [N] -> [3N]
                col_mask = is_free.view(-1, 1).repeat(1, 3).flatten()
                active_indices = torch.nonzero(col_mask).squeeze()
                if active_indices.ndim == 0: active_indices = active_indices.unsqueeze(0)
            else:
                col_mask = torch.ones(n_atoms * 3, device=device)
                active_indices = torch.arange(n_atoms * 3, device=device)

            # 2. 随机行采样 (Stochastic Row Sampling)
            # 仅在训练模式且指定了采样数时启用
            if training and max_samples is not None and len(active_indices) > max_samples:
                # 随机打乱并取前 max_samples 个
                perm = torch.randperm(len(active_indices), device=device)
                sampled_indices = active_indices[perm[:max_samples]]
            else:
                # 验证/推理时，或者自由度很少时，计算全部
                sampled_indices = active_indices

            # 初始化
            hessian = torch.zeros(3 * n_atoms, 3 * n_atoms, device=device)
            # 记录哪些行是计算过的 (用于 Loss)
            row_mask = torch.zeros(3 * n_atoms, 1, device=device)
            
            if sampled_indices.numel() == 0:
                return hessian, row_mask

            forces_flat = forces.flatten()
            
            # 3. 只循环采样到的行
            # 内存消耗大大降低，只保留 sampled_indices 数量的计算图
            for i in sampled_indices:
                v = torch.zeros_like(forces_flat)
                v[i] = 1.0
                
                grad_row = torch.autograd.grad(
                    outputs=forces_flat,
                    inputs=pos,
                    grad_outputs=v,
                    retain_graph=True,
                    create_graph=training
                )[0]
                
                row_content = grad_row.flatten()
                
                # 填充 Hessian (列遮罩 col_mask 依然生效，过滤掉固定原子对力的贡献)
                hessian[i] = row_content * col_mask
                
                # 标记该行有效
                row_mask[i] = 1.0

            hessian = -hessian
            
            # 同时返回 Hessian 和 行掩码
            return hessian, row_mask
