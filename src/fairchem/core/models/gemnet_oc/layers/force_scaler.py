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
        [增强版] 计算 Hessian 矩阵，添加全面的数值稳定性控制
        """
        # 1. 自动确定模式
        if training is None:
            training = self.training

        pos = data.pos
        n_atoms = pos.shape[0]
        device = pos.device
        n_dofs = 3 * n_atoms
        
        # ==========================================
        # 2. 确定要计算的行 (默认所有自由度)
        # ==========================================
        active_indices = torch.arange(n_dofs, device=device)

        # ==========================================
        # 3. 采样逻辑
        # ==========================================
        if training:
            if max_samples is not None and len(active_indices) > max_samples:
                perm = torch.randperm(len(active_indices), device=device)
                sampled_indices = active_indices[perm[:max_samples]]
            else:
                sampled_indices = active_indices
        else:
            sampled_indices = active_indices

        # 初始化输出容器
        hessian = torch.zeros(n_dofs, n_dofs, device=device)
        row_mask = torch.zeros(n_dofs, 1, device=device)
        
        if sampled_indices.numel() == 0:
            return hessian, row_mask
        
        if not getattr(data, 'hessian_mask', True):
            return hessian, row_mask

        # 标记被选中的行
        row_mask[sampled_indices] = 1.0
        
        # ==========================================
        # 4. 核心计算：添加数值稳定性控制
        # ==========================================
        forces_flat = forces.flatten()
        num_sampled = sampled_indices.numel()
        
        # 数值稳定性参数
        hessian_clamp_min = -100.0  # 防止极小值
        hessian_clamp_max = 100.0   # 防止极大值
        max_gradient_norm = 100.0    # 梯度裁剪阈值
        
        for i in range(num_sampled):
            idx = sampled_indices[i]
            
            # 构造基向量
            v = torch.zeros_like(forces_flat)
            v[idx] = 1.0 
            
            try:
                # 计算单行梯度
                grad = torch.autograd.grad(
                    outputs=forces_flat,
                    inputs=pos,
                    grad_outputs=v,
                    retain_graph=True,
                    create_graph=training
                )[0]
                
                # 关键：数值稳定性控制
                grad_flat = grad.flatten()
                
                # 1. 梯度范数检查和裁剪
                grad_norm = torch.norm(grad_flat)
                if grad_norm > max_gradient_norm:
                    grad_flat = grad_flat / grad_norm * max_gradient_norm
                    logging.warning(f"Hessian row {idx}: Gradient norm {grad_norm:.2f} clipped to {max_gradient_norm}")
                
                # 2. 值域限制
                grad_flat = torch.clamp(grad_flat, min=hessian_clamp_min, max=hessian_clamp_max)
                
                # 3. 异常值检查
                if not torch.all(torch.isfinite(grad_flat)):
                    logging.warning(f"Hessian row {idx}: Contains non-finite values, setting to zero")
                    grad_flat = torch.zeros_like(grad_flat)
                
                # 填入矩阵
                hessian[idx] = grad_flat
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"OOM at Hessian row {idx}, setting to zero")
                    hessian[idx] = torch.zeros(n_dofs, device=device)
                else:
                    logging.error(f"Error computing Hessian row {idx}: {str(e)}")
                    hessian[idx] = torch.zeros(n_dofs, device=device)
            
            # 显存清理
            del v
            if 'grad' in locals():
                del grad
            
            # 定期清理 CUDA 缓存
            if training and (i % 10 == 0):
                torch.cuda.empty_cache()

        # 加上负号 (F = -dH/dx)
        hessian = -hessian
        
        # ==========================================
        # 5. 后处理：对称化和最终检查
        # ==========================================
        # if not training:
        #     # 推理模式：强制对称化
        #     hessian = 0.5 * (hessian + hessian.T)
            
        #     # 最终数值检查和修正
        #     hessian = torch.clamp(hessian, min=hessian_clamp_min, max=hessian_clamp_max)
            
        #     # 检查是否仍有异常值
        #     if not torch.all(torch.isfinite(hessian)):
        #         logging.error("Final Hessian contains non-finite values, applying corrections")
        #         # 修复异常值
        #         hessian = torch.nan_to_num(hessian, nan=0.0, posinf=hessian_clamp_max, neginf=hessian_clamp_min)

        return hessian, row_mask
