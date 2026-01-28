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



    def compute_hessian_masked(self, forces, data, training=False):
        """
        根据已有的 force 张量计算 Hessian。
        只计算 data.tags == 1 的部分（按需计算），极大加速过渡态搜索。
        
        Args:
            forces: 模型输出的力 [N, 3]，必须带有 grad_fn (requires_grad=True 的结果)
            data: 数据对象，包含 data.pos [N, 3] 和 data.tags [N]
                data.pos 必须是 requires_grad=True 且是 forces 的祖先节点
        
        Returns:
            hessian: [3N, 3N] 的矩阵。
                    其中只有 tags=1 对应的行和列有值，其余为 0。
        """
        pos = data.pos
        n_atoms = pos.shape[0]
        
        # 1. 准备 Mask
        # 假设 tags=1 为移动原子 (1), tags=0 为固定原子 (0)
        if hasattr(data, 'tags') and data.tags is not None:
            # [N] -> [N, 1] -> [N, 3] -> flatten [3N]
            # 扩展 mask 到 xyz 三个维度
            mask = data.tags.view(-1, 1).repeat(1, 3).flatten()  # shape: [3N]
            
            # 获取所有需要计算的自由度索引 (Active Indices)
            # 例如: [0, 1, 2, 6, 7, 8...] 对应第1个和第3个原子的xyz
            active_indices = torch.nonzero(mask).squeeze()
        else:
            # 如果没有 tags，计算所有
            active_indices = torch.arange(n_atoms * 3, device=pos.device)
            mask = torch.ones(n_atoms * 3, device=pos.device)

        # 初始化 Hessian 矩阵 (稀疏或稠密均可，这里用稠密矩阵方便后续操作)
        hessian = torch.zeros(3 * n_atoms, 3 * n_atoms, device=pos.device)
        
        # 展平 force 以便索引，注意这里并不切断梯度
        forces_flat = forces.flatten()
        
        # 2. 核心循环：只遍历“活跃”的自由度 (Row-wise computation)
        # 对于 20 个自由原子，循环 60 次；对于 200 个原子，如果只动 20 个，依然只循环 60 次！
        # 这是比计算完整 Jacobian 快得多的原因。
        for i in active_indices:
            # 创建一个 One-hot 向量作为 grad_outputs
            # 这相当于告诉 autograd: "只把 force 向量中第 i 个分量的梯度提取出来"
            v = torch.zeros_like(forces_flat)
            v[i] = 1.0
            
            # 计算 v^T * Jacobian (即 Jacobian 的第 i 行)
            # retain_graph=True 是必须的，因为我们要多次反向传播
            grad_row = torch.autograd.grad(
                outputs=forces_flat,
                inputs=pos,
                grad_outputs=v,
                retain_graph=True,
                create_graph=training # 如果不需要训练 Hessian 本身，设为 False 节省显存
            )[0]
            
            # grad_row 的形状是 [N, 3]，我们需要展平放入 Hessian 矩阵
            # 注意：这里计算出的梯度包含所有原子对第 i 个自由度的贡献 (包括固定原子)
            # 我们根据 mask 再次过滤列 (如果你希望固定原子完全不影响 Hessian)
            row_content = grad_row.flatten()
            
            # 填充 Hessian 的第 i 行
            # * mask 是为了确保列也被 mask 掉 (即忽略固定原子的贡献)
            hessian[i] = row_content * mask

        # 3. 物理转换
        # Force = - dE/dx  =>  Hessian = d^2E/dx^2 = - dF/dx
        hessian = -hessian
        
        return hessian

    # --- 使用示例 ---
    # 假设你在某个优化循环中
    # 1. 准备数据
    # data.pos.requires_grad_(True)
    # 
    # 2. 前向传播 (只做一次)
    # forces = model(data) 
    #
    # 3. 计算 Hessian (直接传入 forces，无需定义函数闭包)
    # H = compute_hessian_masked(forces, data)
