import torch
import torch.nn as nn
# import torch_geometric
from torch_runstats.scatter import scatter
# from typing import Optional
import numpy as np
from ase.data import chemical_symbols
# import jax
# import jax.numpy as jnp
# from jaxopt import LBFGS


# 定义一些常量
ELECTRON_CHARGE = 1.602176634e-19  # 电子电荷 (C)
COULOMB_CONSTANT = 8.9875517923e9  # 库仑常数 (N m^2 C^-2)
ANGSTROM_TO_METER = 1e-10  # 埃到米的转换因子
EV_TO_JOULE = 1.602176634e-19  # 电子伏特到焦耳的转换因子


name2eta = {
    "K":3.84,
    "C":10,
    "H":12.84,
    "O":12.16,
    "Ni":6.48,
    "N":14.6,
}

name2chi = {
    "K":2.42,
    "C":6.26,
    "H":7.18,
    "O":7.54,
    "Ni":4.4,
    "N":7.23,
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
    def initialize_weights(self):
        # linear_layers = []
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         linear_layers.append(m)
        
        # # 对除最后一层外的所有线性层使用 Xavier 初始化
        # for layer in linear_layers[:-1]:
        #     nn.init.xavier_uniform_(layer.weight)
        #     nn.init.constant_(layer.bias, 0)

        # # 单独处理最后一层，偏置设为 0.5
        # if linear_layers:
        #     final_layer = linear_layers[-1]
        #     nn.init.xavier_uniform_(final_layer.weight)
        #     nn.init.constant_(final_layer.bias, 1.0)
        pass
                


class QEqModule(nn.Module):
    def __init__(self, charge_ub=2.0, ele_factor=COULOMB_CONSTANT,
                 coul_damping_beta=18.7, coul_damping_r0=2.2,  # r0 in Angstroms
                 charge_mlp_hidden_dims=[16, 16],
                 electronegativity_mlp_hidden_dims=[16, 16],
                 hardness_mlp_hidden_dims=[16, 16],
                 ewald_p=0.3275911,
                 ewald_a=[0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429],
                 ewald_f=2.0):
        """
        初始化QEq模块
        :param charge_ub: 原子电荷上限 (e)
        :param ele_factor: 静电相互作用因子 (N m^2 C^-2)
        :param coul_damping_beta: 库仑相互作用阻尼参数beta
        :param coul_damping_r0: 库仑相互作用阻尼参数r0 (Å) - 注意单位是埃
        :param charge_mlp_hidden_dims: 电荷预测MLP的隐藏层维度
        :param electronegativity_mlp_hidden_dims: 电负性预测MLP的隐藏层维度
        :param hardness_mlp_hidden_dims: 硬度预测MLP的隐藏层维度
        :param ewald_p: Ewald求和参数p
        :param ewald_a: Ewald求和参数a
        :param ewald_f: Ewald求和参数f
        """
        super(QEqModule, self).__init__()
        self.charge_ub = charge_ub
        self.ele_factor = ele_factor
        self.coul_damping_beta = coul_damping_beta
        # 将r0从埃转换为米，保持单位一致性
        self.coul_damping_r0 = coul_damping_r0 * ANGSTROM_TO_METER  # 转换为米
        self.charge_mlp_hidden_dims = charge_mlp_hidden_dims
        self.electronegativity_mlp_hidden_dims = electronegativity_mlp_hidden_dims
        self.hardness_mlp_hidden_dims = hardness_mlp_hidden_dims
        self.ewald_p = ewald_p
        self.ewald_a = torch.tensor(ewald_a)
        self.ewald_f = ewald_f

        # 定义目标电荷字典作为模块的常量
        self.target_charge_dict = {
            'K': -0.88, 'O': 1.36, 'H': -0.69, 'C': -0.1, 'N': 1.13,
            'Ni': -0.78,
        }

        self.charge_mlp_initialized = False
        self.electronegativity_mlp = MLP(256, self.electronegativity_mlp_hidden_dims, 1).to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        self.hardness_mlp = MLP(256, self.hardness_mlp_hidden_dims, 1).to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        self.charge_mlp = MLP(256+1, self.charge_mlp_hidden_dims, 1).to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        # self.charge_mlp.initialize_weights()
        
        # 添加可训练的电荷偏置参数
        # self.charge_biases = nn.ParameterDict({
        #     elem: nn.Parameter(torch.tensor([bias], dtype=torch.float32))
        #     for elem, bias in self.target_charge_dict.items()
        # })

    def initialize_charge_mlp(self, node_feat, inputs):
        """
        单独的初始化方法，只在训练开始时调用一次
        """
        print("Initializing charge MLP with target values...")
        max_iterations = 20000
        convergence_threshold = 0.15
        learning_rate = 0.01

        with torch.no_grad():
            charge = self.charge_mlp(node_feat).squeeze(-1)
            # charge = self.charge_ub * torch.tanh(charge / self.charge_ub)
            
            element_indices = {}
            for idx, elem in enumerate(inputs['atomic_numbers']):
                elem_symbol = chemical_symbols[int(elem)]
                if elem_symbol not in element_indices:
                    element_indices[elem_symbol] = []
                element_indices[elem_symbol].append(idx)

            for iteration in range(max_iterations):
                max_diff = 0.0
                for elem, indices in element_indices.items():
                    if elem in self.target_charge_dict:
                        current_avg = torch.mean(charge[indices])
                        target_avg = self.target_charge_dict[elem]
                        diff = target_avg - current_avg
                        max_diff = max(max_diff, float(abs(diff)))
                        charge[indices] += learning_rate * diff

                # charge = self.charge_ub * torch.tanh(charge / self.charge_ub)

                if max_diff < convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations")
                    break

            # 更新charge_mlp的最后一层偏置，使其产生所需的初始分布
            with torch.enable_grad():
                initial_output = self.charge_mlp(node_feat).squeeze(-1)
                diff = charge - initial_output
                last_layer = list(self.charge_mlp.mlp.children())[-1]
                last_layer.bias.data += diff.mean()

        self.charge_mlp_initialized = True

        print('charge', charge[inputs.batch == 0])

    def predict_charge(self, node_feat, inputs):
        """
        预测原子电荷，在训练和推理时使用不同的策略
        """
        # if self.training:
        #     if not self.charge_mlp_initialized:
        #         self.initialize_charge_mlp(node_feat, inputs)

        # 获取基础电荷预测
        charge = self.charge_mlp(node_feat).squeeze(-1).float()
        
        # 在训练模式下，添加元素特定的偏置
        # for elem, indices in self._get_element_indices(inputs['atomic_numbers']).items():
        #     if elem in self.charge_biases:
        #         charge[indices] = charge[indices] + self.charge_biases[elem]
        
        # 应用tanh限制
        charge = self.charge_ub * torch.tanh(charge / self.charge_ub)
        
        # 计算电负性，用于加权
        # pred_electronegativity = self.electronegativity_mlp(node_feat).squeeze(-1)

        # en_dict = {
        #     'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04,
        #     'O': 3.44, 'F': 3.98, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90,
        #     'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'K': 0.82, 'Ca': 1.00, 'Sc': 1.36,
        #     'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88,
        #     'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18,
        #     'Se': 2.48, 'Br': 2.96, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
        #     'Nb': 1.59, 'Mo': 2.16, 'Tc': 1.91, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20,
        #     'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.12,
        #     'I': 2.66, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13,
        #     'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.22,
        #     'Dy': 1.23, 'Ho': 1.24, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.26, 'Lu': 1.27,
        #     'Hf': 1.30, 'Ta': 1.50, 'W': 2.36, 'Re': 1.93, 'Os': 2.18, 'Ir': 2.20,
        #     'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02
        # }   
        # pred_electronegativity = torch.tensor([en_dict[chemical_symbols[int(i)]] for i in inputs['atomic_numbers']], device=inputs['atomic_numbers'].device)
        
        # # 基于电负性计算权重 (逆比例，电负性高的原子权重小)
        # epsilon = 1e-6  # 避免除零
        # weights = 1.0 / (pred_electronegativity + epsilon)
        
        # # 计算每个分子的总charge和natoms
        # sum_charge = scatter(charge, inputs.batch, dim=0)
        # natoms = scatter(torch.ones_like(inputs.batch, dtype=torch.float32), inputs.batch, dim=0)
        
        # # 计算每个分子的总权重
        # sum_weights = scatter(weights, inputs.batch, dim=0)
        
        # # 规范化权重 (per-molecule)
        # normalized_weights = weights / torch.gather(sum_weights, 0, inputs.batch)
        
        # # 计算每个分子的总diff
        # total_diff = inputs['charge'] - sum_charge
        
        # # 分配diff到每个原子
        # diff_charge = normalized_weights * torch.gather(total_diff, 0, inputs.batch)
        
        # pred_charge = charge + diff_charge

        return charge

    def _get_element_indices(self, atomic_numbers):
        """
        辅助方法：获取每个元素的原子索引
        """
        element_indices = {}
        for idx, elem in enumerate(atomic_numbers):
            elem_symbol = chemical_symbols[int(elem)]
            if elem_symbol not in element_indices:
                element_indices[elem_symbol] = []
            element_indices[elem_symbol].append(idx)
        return element_indices

    def get_coulomb_energy(self, row, col, dij, pred_charge, g_ewald=None, inputs=None):
        """
        计算库仑相互作用能量
        :param row: 边的起始节点索引
        :param col: 边的结束节点索引
        :param dij: 原子间的相对位置向量 (Å)
        :param pred_charge: 预测的原子电荷 (e)
        :param g_ewald: Ewald求和参数g
        :return: 库仑相互作用能量 (eV) 和力 (eV/Å)
        """
        # 将距离从埃转换为米
        dij_meter = dij * ANGSTROM_TO_METER
        rij = torch.sqrt(torch.sum(torch.square(dij_meter), dim=-1))
        
        # 将 pred_charge 从元电荷 (e) 转换为库仑 (C)
        pred_charge_coulomb = pred_charge * ELECTRON_CHARGE
        
        # 基础库仑相互作用
        prefactor_coul = self.ele_factor * pred_charge_coulomb[row] * pred_charge_coulomb[col] / rij
        
        # 使用更温和的阻尼函数，避免过度抑制
        # 对于原子尺度，使用更小的阻尼参数
        r0_meter = self.coul_damping_r0  # 已经是米单位
        
        # 简化的阻尼函数，减少过度抑制
        # 当距离接近r0时开始阻尼，但不过度
        damp_factor = torch.where(
            rij < r0_meter,
            torch.exp(-self.coul_damping_beta * (r0_meter - rij) / r0_meter),
            torch.ones_like(rij)
        )
        
        # 计算库仑能量，使用阻尼因子
        ecoul = 0.5 * prefactor_coul * damp_factor
        
        # 计算库仑力（用于后续计算）
        fcoul = prefactor_coul * damp_factor / rij

        # compute erfc correction in inference, not available in training
        if g_ewald is not None:
            grij = g_ewald * rij
            expm2 = torch.exp(-grij * grij)
            t = 1.0 / (1.0 + self.ewald_p * grij)
            erfc = t * (self.ewald_a[0] + t * (self.ewald_a[1] + t * (self.ewald_a[2] + t * (self.ewald_a[3] + t * self.ewald_a[4])))) * expm2
            ecoul += prefactor_coul * (erfc - 1.0)
            fcoul += prefactor_coul * (erfc + self.ewald_f * grij * expm2 - 1.0) / rij

        # 计算力向量 (fcoul 单位: N, dij 单位: m)
        # 力的方向：从原子i指向原子j，所以使用 dij 的方向
        coul_fij = dij_meter * (fcoul / rij).unsqueeze(-1)  # 单位: N
        
        # 将力从牛顿转换为 eV/Å
        # 1 N = 1 J/m = 1 eV / (1.602176634e-19 J/eV * 1e-10 m/Å) = 6.242e+18 eV/Å
        # 所以转换因子应该是 ANGSTROM_TO_METER / EV_TO_JOULE
        coul_fij_ev_angstrom = coul_fij * ANGSTROM_TO_METER / EV_TO_JOULE
        
        # 将能量从焦耳转换为电子伏特
        ecoul_ev = ecoul / EV_TO_JOULE

        natoms = pred_charge.size(0)
        # 聚合到分子级别，注意避免重复计算（每对原子只计算一次）
        coul_energy = scatter(scatter(ecoul_ev, row, dim=0, dim_size=natoms), inputs.batch, dim=0, dim_size=inputs.batch.max() + 1)
        # 聚合力，确保行动-反应对称
        force_row = scatter(coul_fij_ev_angstrom, row, dim=0, dim_size=natoms)
        force_col = scatter(coul_fij_ev_angstrom, col, dim=0, dim_size=natoms)
        coul_force = force_row - force_col
        # 调试打印
        # print(f"Coulomb force (post-aggregation) range: {coul_force.min():.3e} - {coul_force.max():.3e} eV/Å")

        return coul_energy, coul_force

    # def get_coulomb_energy_ewald(self, row, col, dij, pred_charge, inputs=None, r_cutoff=6.0, sigma=2.2, accuracy=1e-5):
        """
        计算库仑相互作用能量
        """
        # self.real_space = 0
        # self.reciprocal_space = 0
        # self.self_energy = 0
        # self.corr_energy = 0

        # dij_meter = dij * ANGSTROM_TO_METER
        # # eta = torch.sqrt(-torch.log(accuracy)) / r_cutoff
        # eta = torch.sqrt(-torch.log(accuracy)) / r_cutoff
        # kmax = torch.ceil(2 * eta * torch.sqrt(- torch.log(accuracy))).int()


        # a = torch.erfc(dij_meter * eta/ dij_meter)
        # b = a * pred_charge[row] * pred_charge[col]
        # self.real_space += torch.sum(b) * 0.5

        # self.self_energy += -torch.sum(pred_charge ** 2 / torch.sqrt(torch.pi) * eta) 

        # k = torch.arange(-kmax, kmax+1)
        # kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        # k_vecs = 2 * np.pi * torch.stack([kx, ky, kz], dim=-1) / inputs.cell
        # V = torch.prod(inputs.cell)
        # k_vecs = k_vecs.reshape(-1, 3)
        # c = (torch.exp(1j * k_vecs @ inputs.pos.T) @ pred_charge)**2 
        # d = torch.exp(-torch.norm(k_vecs, dim=-1)**2 / (4 * eta**2)) / torch.norm(k_vecs, dim=-1)**2 / V / (2 * torch.pi)

        # self.reciprocal_space += torch.sum(c * d)

        # self.corr_energy += -torch.pi * torch.sum(pred_charge) ** 2 / ( 2* eta**2 * V)

    def get_coulomb_energy_ewald(self, row, col, dij, pred_charge, inputs=None, r_cutoff=6.0, accuracy=1e-5):
        """
        完全基于α参数的Ewald求和计算（四项能量）
        """
        # 初始化能量项
        self.real_space = 0.0
        self.reciprocal_space = 0.0
        self.self_energy = 0.0
        self.corr_energy = 0.0

        # 单位转换和参数计算（仅使用α）
        dij_meter = dij * ANGSTROM_TO_METER
        alpha = torch.sqrt(-torch.log(torch.tensor(accuracy))) / r_cutoff  # 直接定义α
        
        # 非正交晶胞体积和倒易基矢
        cell = inputs.cell  # [3,3]张量，每行是一个晶胞向量
        V = torch.abs(torch.det(cell))  # 行列式计算体积
        B = 2 * torch.pi * torch.linalg.inv(cell).T  # 倒易基矢矩阵[3,3]

        # 1. 实空间项：erfc(αr_ij)/r_ij
        mask = (dij_meter > 1e-10) & (dij_meter < r_cutoff)
        dij_safe = dij_meter[mask] + 1e-10
        erfc_term = torch.erfc(alpha * dij_safe) / dij_safe  # 使用α
        self.real_space = 0.5 * torch.sum(pred_charge[row[mask]] * pred_charge[col[mask]] * erfc_term)

        # 2. 自能项：-α/√π Σq_i²
        self.self_energy = - (alpha / torch.sqrt(torch.pi)) * torch.sum(pred_charge**2)

        # 3. 倒易空间项：1/(2πV) Σ e^{-k²/(4α²)}/k² |S(k)|²
        kmax = torch.ceil(2 * alpha * torch.sqrt(-torch.log(torch.tensor(accuracy)))).int()
        k = torch.arange(-kmax, kmax+1, device=dij_meter.device)
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        k_grid = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)  # [n_k,3]整数网格
        k_vecs = k_grid @ B.T  # 转换为真实k矢量[3,3]
        
        # 排除k=0
        k_norms = torch.norm(k_vecs, dim=1)
        non_zero_mask = k_norms > 1e-10
        k_vecs = k_vecs[non_zero_mask]
        k_norms = k_norms[non_zero_mask]

        # 计算结构因子S(k)
        k_dot_r = torch.matmul(k_vecs, inputs.pos.T)  # [n_k, n_atoms]
        S_k = torch.matmul(torch.exp(1j * k_dot_r), pred_charge)  # [n_k]
        
        # 倒易空间能量系数
        coeff = torch.exp(-k_norms**2 / (4 * alpha**2)) / (k_norms**2 * V * 2 * torch.pi)  # 使用α²
        self.reciprocal_space = torch.sum(torch.abs(S_k)**2 * coeff)

        # 4. 表面校正项：-πQ²/(2α²V)
        Q_tot = torch.sum(pred_charge)
        if abs(Q_tot) > 1e-10:
            self.corr_energy = - (torch.pi * Q_tot**2) / (2 * alpha**2 * V)  # 使用α²

        # 总库仑能
        total_energy = self.real_space + self.reciprocal_space + self.self_energy + self.corr_energy
        return total_energy






    def get_qeq_force(self, charge_energy, inputs, row, col, natoms, grad_outputs=None):
        """
        通过自动微分计算电荷能量对位置的梯度，得到力
        :param charge_energy: 电荷能量 (eV)
        :param inputs: 包含位置向量的输入数据
        :param row, col: 边的索引
        :param natoms: 原子数量
        :param grad_outputs: 梯度输出
        :return: 力 (eV/Å)
        """
        # charge_energy 单位: eV, inputs['vector'] 单位: Å
        # 自动微分 dE/dr 得到力，单位: eV/Å
        qeq_fij = torch.autograd.grad([charge_energy], [inputs['vector']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] 
        if qeq_fij is None: # used for torch.jit.script
            qeq_fij_cast = torch.zeros(size=inputs['vector'].size(), device=inputs['vector'].device)
        else:
            qeq_fij_cast = -1.0 * qeq_fij  # 力是能量的负梯度
        qeq_force = scatter(qeq_fij_cast, row, dim=0, dim_size=natoms) - scatter(qeq_fij_cast, col, dim=0, dim_size=natoms)
        return qeq_force
    # def get_coulomb_force(self, row, col, dij, pred_charge, g_ewald=None):
    #     """
    #     计算库仑相互作用的力
    #     :param row: 边的起始节点索引
    #     :param col: 边的结束节点索引
    #     :param dij: 原子间的相对位置向量 (Å)
    #     :param pred_charge: 预测的原子电荷 (e)
    #     :param g_ewald: Ewald求和参数g
    #     :return: 库仑相互作用力 (eV/Å)
    #     """
    #     # 将距离从埃转换为米
    #     dij_meter = dij * ANGSTROM_TO_METER
    #     rij = torch.sqrt(torch.sum(torch.square(dij_meter), dim=-1))
    #     # 将 pred_charge 从元电荷 (e) 转换为库仑 (C)
    #     pred_charge_coulomb = pred_charge * ELECTRON_CHARGE
    #     prefactor_coul = self.ele_factor * pred_charge_coulomb[row] * pred_charge_coulomb[col] / rij
    #     damp_coul = torch.sigmoid(self.coul_damping_beta / self.coul_damping_r0 * (rij - self.coul_damping_r0))
    #     softplus_coul = nn.functional.softplus((rij - self.coul_damping_r0) / self.coul_damping_r0)
    #     r0 = self.coul_damping_r0
    #     fcoul = prefactor_coul * damp_coul * (rij / r0 / (1 + softplus_coul))**2

    #     if g_ewald is not None:
    #         grij = g_ewald * rij
    #         expm2 = torch.exp(-grij * grij)
    #         t = 1.0 / (1.0 + self.ewald_p * grij)
    #         erfc = t * (self.ewald_a[0] + t * (self.ewald_a[1] + t * (self.ewald_a[2] + t * (self.ewald_a[3] + t * self.ewald_a[4])))) * expm2
    #         fcoul += prefactor_coul * (erfc + self.ewald_f * grij * expm2 - 1.0)

    #     # 计算库仑力向量
    #     coul_fij = dij * (fcoul / rij / rij).unsqueeze(-1)

    #     # 将力从牛顿转换为电子伏特每埃
    #     coul_fij_ev_angstrom = coul_fij * ANGSTROM_TO_METER / EV_TO_JOULE

    #     natoms = pred_charge.size(0)
    #     coul_force = scatter(coul_fij_ev_angstrom, row, dim=0, dim_size=natoms) - scatter(coul_fij_ev_angstrom, col, dim=0, dim_size=natoms)

    #     return coul_force
 

    def get_electronegativity_energy(self, pred_electronegativity, pred_electronegativity_hardness, pred_charge, inputs, nmols):
        """
        计算电负性能量
        :param node_feat: 节点特征
        :param pred_charge: 预测的原子电荷 (e)
        :param inputs: 输入数据，包含mol_ids
        :return: 电负性能量 (eV)
        """

        
        # 电负性能量使用原子单位计算，pred_charge 保持元电荷单位 (e)
        # 电负性能量公式：E = χ² * q + η² * q²，其中 χ 是电负性，η 是硬度，q 是电荷
        # 为了得到合理的能量量级，需要适当的缩放因子
        scale_factor = 1.0  # 可以根据需要调整
        
        electronegativity_energy = scale_factor * (
            pred_electronegativity * pred_charge + 
            0.5 * pred_electronegativity_hardness * pred_charge * pred_charge
        )
        
        # 聚合到分子级别
        electronegativity_energy = scatter(electronegativity_energy, inputs.batch, dim=0, dim_size=nmols)
        
        # 电负性能量已经是 eV 量级，不需要额外转换
        return electronegativity_energy

    def get_electronegativity(self, node_feat):
        """
        计算电负性
        :param node_feat: 节点特征
        :param inputs: 输入数据，包含mol_ids
        :return: 电负性
        """
        pred_electronegativity = self.electronegativity_mlp(node_feat).squeeze(-1)
        pred_electronegativity_hardness = self.hardness_mlp(node_feat).squeeze(-1)
        # name2chi.get(data.atomic_number)
        # pred_electronegativity = torch.tensor([name2chi[chemical_symbols[int(i)]] for i in inputs['atomic_numbers']], device=inputs['atomic_numbers'].device)
        # pred_electronegativity_hardness = torch.tensor([name2eta[chemical_symbols[int(i)]] for i in inputs['atomic_numbers']], device=inputs['atomic_numbers'].device)
        return pred_electronegativity, pred_electronegativity_hardness

    def get_charge_energy(self, node_feat, row, col, dij, inputs, g_ewald=None):
        """
        计算总电荷能量
        :param node_feat: 节点特征
        :param row: 边的起始节点索引
        :param col: 边的结束节点索引
        :param dij: 原子间的相对位置向量 (Å)
        :param inputs: 输入数据，包含mol_ids和total_charge
        :param g_ewald: Ewald求和参数g
        :return: 总电荷能量 (eV)
        """
        pred_charge = self.predict_charge(node_feat, inputs)
        coul_energy, _ = self.get_coulomb_energy(row, col, dij, pred_charge, g_ewald)
        coul_energy = 0.5 * scatter(scatter(coul_energy, inputs['edge_index'][0], dim=0, dim_size=len(inputs.batch)), inputs.batch, dim=0, dim_size=inputs.batch.max() + 1)
        electronegativity_energy = self.get_electronegativity_energy(node_feat, pred_charge, inputs, nmols=inputs.batch.max() + 1)
        charge_energy = coul_energy + electronegativity_energy
        return charge_energy

    def forward(self, node_feat, pos, batch, Q_total, edge_index=None, edge_weight=None):
        """
        QEqModule 的前向传播，整合预测和求解。
        """
        # 1. 预测电负性 chi 和硬度 eta
        chi, eta = self.predict_charge(node_feat, None) # inputs 可能不需要
        # 2. 求解 QEq 方程得到 q_eq
        q_eq = self.solve_qeq_linear_system(chi, eta, pos, batch, Q_total, edge_index, edge_weight)
        return q_eq, chi, eta

    def solve_qeq_linear_system(self, chi, eta, pos, batch, Q_total, edge_index=None, edge_weight=None):
        """
        使用矩阵求逆法 (实际上是求解线性方程组 Ax=b) 来求解 QEq 方程。

        关键概念:
        - 我们想要找到一组原子电荷 q_eq，使得由这些电荷产生的能量最小，并且满足体系总电荷约束 sum(q_eq) = Q_total。
        - 这个问题可以通过求解一个增广的线性方程组来解决。
        - 增广矩阵 A_aug 和向量 b_aug 构成了这个方程组: A_aug * [q_eq; lambda] = b_aug
            其中 lambda 是拉格朗日乘子，用来强制执行电荷守恒约束。

        Args:
            chi (torch.Tensor): [num_atoms] 预测的每个原子的电负性 (描述原子吸引电子的趋势)。
            eta (torch.Tensor): [num_atoms] 预测的每个原子的化学硬度 (描述原子抵抗电荷变化的能力)。
            pos (torch.Tensor): [num_atoms, 3] 原子的笛卡尔坐标 (单位: Angstrom)。
            batch (torch.Tensor): [num_atoms] 每个原子属于哪个分子/体系的索引。
                                    例如: [0, 0, 0, 1, 1] 表示前3个原子属于体系0，后2个属于体系1。
            Q_total (torch.Tensor): [num_systems] 每个体系的目标总电荷。
            edge_index (torch.Tensor, optional): [2, num_edges] 邻接表，指定哪些原子对之间有相互作用。
                                                edge_index[0] 是源原子索引，edge_index[1] 是目标原子索引。
                                                如果为 None，函数会为每个体系内的原子构建全连接图（效率较低）。
            edge_weight (torch.Tensor, optional): [num_edges] 每条边的权重，通常是库仑核 1/r_ij。
                                                    如果为 None，函数会根据 pos 和 edge_index 计算。

        Returns:
            q_eq (torch.Tensor): [num_atoms] 求解得到的符合约束的原子电荷。
        """
        device = pos.device
        dtype = pos.dtype
        num_atoms = pos.shape[0]                        # 总原子数
        num_systems = Q_total.shape[0]                  # 总体系数
        EV_TO_JOULE = 1.602176634e-19                   # eV 到 Joule 的转换因子
        ANGSTROM_TO_METER = 1e-10                       # Angstrom 到 Meter 的转换因子
        COULOMB_CONSTANT = 8.9875517923e9              # 库仑常数 (N m^2 C^-2)

        # 2. 构建邻接表 (如果未提供)
        #    邻接表定义了哪些原子对之间存在库仑相互作用。
        #    如果没有提供，一个简单但低效的方法是为每个体系构建一个全连接图。
        if edge_index is None:
            # --- 为每个体系构建全连接图 ---
            mol_atom_indices = [] # 存储每个体系的原子全局索引
            for sys_idx in range(num_systems):
                atom_indices_in_sys = torch.where(batch == sys_idx)[0] # 找到属于体系 sys_idx 的所有原子的索引
                mol_atom_indices.append(atom_indices_in_sys)

            all_rows = []
            all_cols = []
            for atom_indices_in_sys in mol_atom_indices:
                n_atoms_in_sys = atom_indices_in_sys.shape[0]
                if n_atoms_in_sys > 1: # 至少需要两个原子才能形成边
                    # 使用 torch.meshgrid 生成体系内所有原子对的索引
                    # indexing='ij' 确保了正确的笛卡尔积顺序
                    rows_sys, cols_sys = torch.meshgrid(atom_indices_in_sys, atom_indices_in_sys, indexing='ij')
                    # 只保留上三角部分 (i < j) 来避免重复边 (i-j 和 j-i 是同一条边)
                    # 并且排除自环 (i == j)
                    mask_upper_triangular = rows_sys < cols_sys
                    all_rows.append(rows_sys[mask_upper_triangular])
                    all_cols.append(cols_sys[mask_upper_triangular])

            if all_rows:
                # 将所有体系的边索引连接起来
                edge_index = torch.stack([torch.cat(all_rows), torch.cat(all_cols)], dim=0) # [2, total_num_edges]
            else:
                # 如果没有任何边 (例如所有体系都只有0或1个原子)
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device) # [2, 0]

        # 3. 计算边权重 (库仑核 1/r) (如果未提供)
        #    边权重通常是 1/r_ij，代表原子 i 和 j 之间的库仑相互作用强度 (忽略电荷)。
        if edge_weight is None and edge_index.shape[1] > 0:
            # 获取边连接的原子坐标
            pos_i = pos[edge_index[0]]  # [num_edges, 3] 源原子坐标
            pos_j = pos[edge_index[1]]  # [num_edges, 3] 目标原子坐标
            # 计算相对位移向量
            rij_vec = pos_i - pos_j     # [num_edges, 3]
            # 计算欧几里得距离 (单位: Angstrom)
            rij_angstrom = torch.norm(rij_vec, dim=1) # [num_edges]
            # 转换为米 (SI单位)
            rij_meter = rij_angstrom * ANGSTROM_TO_METER # [num_edges]
            # 避免除以零或非常小的距离
            rij_meter = torch.clamp(rij_meter, min=1e-12)
            # 计算库仑核 k_e / r_ij (单位: N*m^2/C^2 / m = N*m/C^2)
            # 注意：这里的 k_e / r_ij 还没有乘以 q_i * q_j
            coulomb_kernel = COULOMB_CONSTANT / rij_meter # [num_edges]
            edge_weight = coulomb_kernel # [num_edges]
        elif edge_index.shape[1] == 0:
            # 如果没有边，边权重也为空
            edge_weight = torch.empty(0, dtype=dtype, device=device) # [0]

        # 4. 为每个体系分别求解 QEq 方程
        #    因为不同体系之间没有相互作用，所以可以独立求解。
        q_eq_solutions = [] # 用于存储每个体系求解出的电荷
        num_edges_total = edge_index.shape[1]

        # 遍历每一个体系
        for sys_idx in range(num_systems):

            # --- a. 确定当前体系的原子 ---
            atom_indices_in_sys = torch.where(batch == sys_idx)[0] # [n_atoms_in_this_system]
            n_atoms_in_sys = atom_indices_in_sys.shape[0]          # 当前体系的原子数

            # 如果体系中没有原子，则跳过
            if n_atoms_in_sys == 0:
                continue

            # --- b. 提取当前体系的相关数据 ---
            mol_chi = chi[atom_indices_in_sys]          # [n_atoms_in_sys] 当前体系原子的电负性
            mol_eta = eta[atom_indices_in_sys]          # [n_atoms_in_sys] 当前体系原子的硬度
            mol_Q_total = Q_total[sys_idx]              # 标量，当前体系的目标总电荷

            # --- c. 提取当前体系内部的边 (邻居) ---
            #     找到所有连接当前体系内原子的边
            #     edge_index 中的源和目标原子索引都必须在 atom_indices_in_sys 内
            if num_edges_total > 0:
                # 创建一个掩码，标记哪些边连接的是当前体系内的原子
                mask_edge_in_sys = (edge_index[0, :] >= atom_indices_in_sys[0]) & \
                                    (edge_index[0, :] <= atom_indices_in_sys[-1]) & \
                                    (edge_index[1, :] >= atom_indices_in_sys[0]) & \
                                    (edge_index[1, :] <= atom_indices_in_sys[-1])
                # 获取当前体系内的边索引（全局索引）
                mol_edge_index_global = edge_index[:, mask_edge_in_sys] # [2, num_edges_in_this_mol]
                # 将全局索引转换为当前体系内的局部索引 (0 到 n_atoms_in_sys-1)
                mol_edge_index_local = mol_edge_index_global - atom_indices_in_sys[0] # [2, num_edges_in_this_mol]
                # 获取当前体系内边的权重
                mol_edge_weight = edge_weight[mask_edge_in_sys]       # [num_edges_in_this_mol]
            else:
                mol_edge_index_local = torch.empty((2, 0), dtype=torch.long, device=device) # [2, 0]
                mol_edge_weight = torch.empty(0, dtype=dtype, device=device) # [0]

            num_edges_in_sys = mol_edge_index_local.shape[1]

            # --- d. 构造增广矩阵 A_aug 和向量 b_aug ---
            #     对于 N 个原子，A_aug 是 (N+1) x (N+1)，b_aug 是 (N+1)
            N = n_atoms_in_sys
            # 初始化矩阵和向量
            A_aug = torch.zeros((N + 1, N + 1), dtype=dtype, device=device) # [(N+1), (N+1)]
            b_aug = torch.zeros(N + 1, dtype=dtype, device=device)           # [N+1]

            # -- 填充 A_aug --
            # 1. 对角线部分 A[i,i] = eta[i] (化学硬度)
            #    这代表原子 i 本身抵抗电荷变化的能量代价
            A_aug[:N, :N].fill_diagonal_(0) # (可选) 先清零对角线
            A_aug[:N, :N].diagonal().copy_(mol_eta) # 填充对角线元素

            # 2. 非对角线部分 A[i,j] = A[j,i] = k_e / r_ij (库仑相互作用)
            #    这代表原子 i 和 j 之间的静电排斥能
            if num_edges_in_sys > 0:
                # 获取边的源和目标局部索引
                i_local, j_local = mol_edge_index_local[0, :], mol_edge_index_local[1, :] # [num_edges_in_sys]
                w = mol_edge_weight # [num_edges_in_sys] 即 k_e / r_ij
                # 由于矩阵是对称的，同时填充 A[i,j] 和 A[j,i]
                A_aug[i_local, j_local] = w
                A_aug[j_local, i_local] = w

            # 3. 约束部分: 最后一行和最后一列用于实现 sum(q) = Q_total
            #    A[N, 0:N] = 1  -> 1*q0 + 1*q1 + ... + 1*q(N-1) = Q_total
            #    A[0:N, N] = 1  -> 同上 (矩阵对称性)
            #    A[N, N] = 0    -> 拉格朗日乘子项系数为0
            A_aug[-1, :N] = 1.0      # 填充最后一行的前N列
            A_aug[:N, -1] = 1.0      # 填充最后N行的最后一列
            diag_indices = torch.arange(N)
            A_aug[diag_indices, diag_indices] += eta[atom_indices_in_sys]  # 对角线元素乘以2
            # A_aug[N, N] 默认为 0.0

            # -- 填充 b_aug --
            # b_aug[0:N] = -chi[0:N]  -> 负的电负性作为驱动力
            # b_aug[N] = Q_total      -> 约束值
            b_aug[:N] = -mol_chi           # [N]
            b_aug[N] = mol_Q_total         # 标量

            # --- e. 求解线性方程组 A_aug * x_aug = b_aug ---
            #     x_aug = [q0, q1, ..., q(N-1), lambda]
            #     我们只需要前 N 个解，即原子电荷 q_eq
            try:
                # 使用 LU 分解求解稠密线性系统 (推荐)
                # torch.linalg.solve 返回 x 使得 Ax = b
                x_aug_solution = torch.linalg.solve(A_aug, b_aug.unsqueeze(1)) # solve 需要列向量
                mol_q_eq = x_aug_solution[:N, 0] # 取解向量的前 N 个元素作为电荷 [N]
                lambda_sol = x_aug_solution[N, 0] # (通常不需要拉格朗日乘子)

            except torch.linalg.LinAlgError as e:
                # 如果矩阵是奇异的 (不可逆)，则使用伪逆 (SVD) 来求解 (更鲁棒)
                print(f"Warning: Singular matrix for system {sys_idx}. Using pseudo-inverse.")
                # 计算 A_aug 的伪逆
                A_pinv = torch.linalg.pinv(A_aug)
                # 用伪逆求解 x_aug = A_pinv * b_aug
                x_aug_solution = A_pinv @ b_aug.unsqueeze(1)
                mol_q_eq = x_aug_solution[:N, 0] # [N]

            # 将当前体系的解存入列表
            q_eq_solutions.append(mol_q_eq)

        # 5. 合并所有体系的解
        #    将所有体系的 q_eq 向量按原子在全局列表中的顺序连接起来
        if q_eq_solutions:
            # torch.cat 在第0维 (原子数维度) 连接
            q_eq = torch.cat(q_eq_solutions, dim=0) # [num_atoms]
        else:
            # 如果没有任何原子 (理论上不应该发生)，返回空张量
            q_eq = torch.empty(0, dtype=dtype, device=device) # [0]

        # 返回最终的原子电荷 q_eq
        return q_eq # [num_atoms]