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
            [方案 3 全量版] 串行计算 Hessian，无视 Fixed 约束，计算全原子相互作用。
            依旧保留手动 GC 以防止 OOM。
            """
            # 1. 自动确定模式
            if training is None:
                training = self.training

            pos = data.pos
            n_atoms = pos.shape[0]
            device = pos.device
            n_dofs = 3 * n_atoms
            
            # ==========================================
            # 2. 确定要计算的行 (不再区分 Fixed/Free)
            # ==========================================
            # 默认所有自由度都是活跃的
            active_indices = torch.arange(n_dofs, device=device)

            # ==========================================
            # 3. 采样逻辑 (Sampling Logic)
            # ==========================================
            if training:
                if max_samples is not None and len(active_indices) > max_samples:
                    # 训练时：随机采样任意行（包括底部原子的行）
                    perm = torch.randperm(len(active_indices), device=device)
                    sampled_indices = active_indices[perm[:max_samples]]
                else:
                    sampled_indices = active_indices
            else:
                # 推理时：计算全量
                sampled_indices = active_indices

            # 初始化输出容器
            hessian = torch.zeros(n_dofs, n_dofs, device=device)
            row_mask = torch.zeros(n_dofs, 1, device=device)
            
            if sampled_indices.numel() == 0:
                return hessian, row_mask
            
            # 标记被选中的行
            row_mask[sampled_indices] = 1.0
            
            # ==========================================
            # 4. 核心计算：串行循环 + 激进的显存清理
            # ==========================================
            forces_flat = forces.flatten()
            
            num_sampled = sampled_indices.numel()
            
            for i in range(num_sampled):
                idx = sampled_indices[i]
                
                # A. 构造基向量 (Basis Vector)
                v = torch.zeros_like(forces_flat)
                v[idx] = 1.0 
                
                # B. 计算单行梯度
                grad = torch.autograd.grad(
                    outputs=forces_flat,
                    inputs=pos,
                    grad_outputs=v,
                    retain_graph=True,    # 必须保留图给下一次迭代
                    create_graph=training # 训练时需要二阶导图
                )[0]
                
                # C. 填入矩阵
                # [修改点]: 去掉了 * col_mask，保留所有列的相互作用
                # 现在的 hessian 是全稠密的 (Full Dense)
                hessian[idx] = grad.flatten()
                
                # D. [关键] 显存清理 (Garbage Collection)
                del v, grad
                
                # E. 强制清空 CUDA 缓存
                if training and (i % 5 == 0):
                    torch.cuda.empty_cache()

            # 加上负号 (F = -dH/dx)
            hessian = -hessian
            
            # ==========================================
            # 5. 推理模式专用后处理
            # ==========================================
            # 注意：这里是你策略的关键。
            # 你选择在模型输出时不强制对称化，把对称化留给外部处理，这没问题。
            # if not training:
            #     hessian = 0.5 * (hessian + hessian.T)

            return hessian, row_mask
