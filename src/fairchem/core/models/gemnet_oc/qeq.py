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

    def get_coulomb_energy_ewald(self, row, col, dij_vec, pred_charge, inputs=None, r_cutoff=6.0, accuracy=1e-5):
        """
        使用α参数的Ewald求和计算（四项能量）
        使用埃（Å）作为基本单位，避免复数运算
        """
        # 初始化能量项
        self.real_space = 0.0
        self.reciprocal_space = 0.0
        self.self_energy = 0.0
        self.corr_energy = 0.0

        # 单位转换常数 (使用Å单位)
        EV_ANGSTROM = 14.399645  # 转换因子: (e²/(4πε₀)) in eV·Å

        # 直接使用Å单位
        r_cutoff_ang = r_cutoff

        # 计算距离 (从矢量计算)
        dij_ang = torch.norm(dij_vec, dim=1)  # [num_edges]

        # 计算α参数 (单位: 1/Å)
        alpha = torch.sqrt(-torch.log(torch.tensor(accuracy))) / r_cutoff_ang

        # 晶胞处理 (单位: Å)
        cell = inputs.cell.view(3, 3)
        V = torch.abs(torch.det(cell))  # 体积 (Å³)
        
        # 计算倒易基矢 (1/Å)，注意这里不需要2π因子
        B = torch.linalg.inv(cell).T

        # 1. 实空间项 (只考虑在截断半径内的原子对)
        mask = (dij_ang > 1e-11) & (dij_ang < r_cutoff_ang)
        dij_safe = dij_ang[mask] + 1e-11
        erfc_term = torch.erfc(alpha * dij_safe) / dij_safe  # 使用α
        
        # 只计算符合条件的原子对
        row_masked = row[mask]
        col_masked = col[mask]
        self.real_space = 0.5 * torch.sum(
            pred_charge[row_masked] * pred_charge[col_masked] * erfc_term
        )

        # 2. 自能项 (使用α)
        self.self_energy = - (alpha / torch.sqrt(torch.tensor(torch.pi).to(torch.float32))) * torch.sum(pred_charge**2)

        # 3. 倒易空间项 (使用α参数和K²计算，避免复数运算)
        # 计算k网格
        kmax = torch.ceil(2 * alpha * torch.sqrt(-torch.log(torch.tensor(accuracy)))).int()
        k = torch.arange(-kmax, kmax+1, device=dij_ang.device)
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        k_grid = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3).to(torch.float32)
        
        # 计算k矢量 (1/Å)，注意这里不需要2π因子
        k_vecs = k_grid @ B.T
        
        # 排除k=0
        k_norms = torch.norm(k_vecs, dim=1)
        non_zero_mask = k_norms > 1e-11
        k_vecs = k_vecs[non_zero_mask]
        k_norms = k_norms[non_zero_mask]
        k_norms_sq = k_norms**2  # K²计算

        # 计算结构因子 S(k) 的模平方，避免复数运算
        k_dot_r = torch.matmul(k_vecs, inputs.pos.T)  # [n_k, n_atoms]
        
        # 使用三角函数计算实部和虚部
        cos_kr = torch.cos(k_dot_r)  # [n_k, n_atoms]
        sin_kr = torch.sin(k_dot_r)  # [n_k, n_atoms]
        
        # 计算实部和虚部的和
        real_sum = torch.matmul(cos_kr, pred_charge)  # [n_k]
        imag_sum = torch.matmul(sin_kr, pred_charge)  # [n_k]
        
        # 计算模平方 |S(k)|² = (实部和)² + (虚部和)²
        S_k_sq = real_sum**2 + imag_sum**2  # [n_k]

        # 按照图片中的公式计算倒易空间能量，但使用α参数
        # 根据 η² = 1/(2α²) 的关系，将公式中的 η 替换为 α
        # 原公式: exp(-η²|k|²/2) = exp(-(1/(2α²))|k|²/2) = exp(-|k|²/(4α²))
        exp_term = torch.exp(-k_norms_sq / (4 * alpha**2))
        coeff = (2 * torch.pi / V) * exp_term / k_norms_sq
        self.reciprocal_space = torch.sum(S_k_sq * coeff)

        # 4. 表面校正项 (使用α)
        Q_tot = torch.sum(pred_charge)
        if abs(Q_tot) > 1e-11:
            # 根据 η² = 1/(2α²) 的关系，将公式中的 η 替换为 α
            # 原公式: - (π Q_tot²) / (2 η² V) = - (π Q_tot²) / (2 * (1/(2α²)) V) = - (π α² Q_tot²) / V
            self.corr_energy = - (torch.pi * alpha**2 * Q_tot**2) / V

        # 总能量 (单位: e²/Å)
        total_energy = self.real_space + self.reciprocal_space + self.self_energy + self.corr_energy

        # 单位转换: e²/Å → eV
        total_energy *= EV_ANGSTROM

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

    def solve_qeq_linear_system(self, chi, eta, inputs, batch, Q_total, edge_index=None, edge_vec=None):
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
        pos = inputs.pos
        device = pos.device
        dtype = pos.dtype
        num_atoms = pos.shape[0]                        # 总原子数
        num_systems = Q_total.shape[0]                  # 总体系数
        EV_TO_JOULE = 1.602176634e-19                   # eV 到 Joule 的转换因子
        ANGSTROM_TO_METER = 1e-10                       # Angstrom 到 Meter 的转换因子
        COULOMB_CONSTANT = 8.9875517923e9              # 库仑常数 (N m^2 C^-2)

        def build_qeq_matrix_A(eta, pos, cell, r_cutoff=6.0, accuracy=1e-5, edge_index=None, edge_vec=None):
            """
            正确构造QEq方程的矩阵A，考虑周期性边界条件
            
            Args:
                eta (torch.Tensor): [num_atoms] 化学硬度 J_i
                pos (torch.Tensor): [num_atoms, 3] 原子坐标
                cell (torch.Tensor): [3, 3] 晶胞向量
                r_cutoff, accuracy: Ewald参数
                edge_index (torch.Tensor): [2, num_edges] 边的索引
                edge_vec (torch.Tensor): [num_edges, 3] 边矢量（考虑周期性）
            
            Returns:
                A_aug (torch.Tensor): [num_atoms + 1, num_atoms + 1] 增广矩阵
            """
            
            device = pos.device
            dtype = pos.dtype
            num_atoms = pos.shape[0]
            
            # 初始化矩阵A (不包括约束)
            A = torch.zeros((num_atoms, num_atoms), dtype=dtype, device=device)

            # --- 1. 计算Ewald参数 ---
            alpha = torch.sqrt(-torch.log(torch.tensor(accuracy, device=device))) / r_cutoff
            V = torch.abs(torch.det(cell))
            B = 2 * torch.pi * torch.linalg.inv(cell).T.to(dtype)

            # --- 2. 实空间项 (只影响非对角线元素) ---
            # 使用边矢量计算距离（考虑周期性）
            if edge_index is not None and edge_vec is not None:
                row, col = edge_index
                dij = torch.norm(edge_vec, dim=1)  # 考虑周期性的距离
                
                # 只处理距离在有效范围内的边
                mask = (dij > 1e-10) & (dij < r_cutoff)
                dij_safe = dij[mask] + 1e-10
                erfc_term = torch.erfc(alpha * dij_safe) / dij_safe
                
                # 填充到矩阵A（对称矩阵）
                A[row[mask], col[mask]] = erfc_term
                A[col[mask], row[mask]] = erfc_term
            else:
                # 如果没有提供边信息，使用全连接（效率较低）
                for i in range(num_atoms):
                    for j in range(i+1, num_atoms):  # 只计算上三角
                        # 计算考虑周期性的最小距离
                        delta = pos[j] - pos[i]
                        # 将距离映射到[-0.5, 0.5]晶胞范围内
                        frac_delta = torch.linalg.solve(cell.T, delta)
                        frac_delta = frac_delta - torch.round(frac_delta)
                        min_delta = frac_delta @ cell
                        dij = torch.norm(min_delta)
                        
                        if dij > 1e-10 and dij < r_cutoff:
                            dij_safe = dij + 1e-10
                            erfc_term = torch.erfc(alpha * dij_safe) / dij_safe
                            A[i, j] = erfc_term
                            A[j, i] = erfc_term

            # --- 3. 自能项和倒易空间项 (只影响对角线元素) ---
            # 计算倒易空间项的对角线部分
            kmax = torch.ceil(2 * alpha * torch.sqrt(-torch.log(torch.tensor(accuracy, device=device)))).int()
            k = torch.arange(-kmax, kmax + 1, device=device)
            kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
            k_grid = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3).to(torch.float32)
            k_vecs = k_grid @ B.T
            k_norms = torch.norm(k_vecs, dim=1)
            non_zero_mask = k_norms > 1e-10
            k_vecs = k_vecs[non_zero_mask]
            k_norms = k_norms[non_zero_mask]

            coeff = torch.exp(-k_norms**2 / (4 * alpha**2)) / (k_norms**2 * V * torch.pi)
            
            # 倒易空间项只影响对角线元素
            reciprocal_diag = torch.sum(coeff)  # 对所有k点求和
            
            # 自能项
            self_energy = -alpha / torch.sqrt(torch.tensor(torch.pi, dtype=dtype))
            
            # 将对角线元素设置为: A_ii = 自能项 + 倒易空间项
            A.diagonal().copy_(self_energy + reciprocal_diag)
            
            # --- 4. 化学硬度项 (根据文献公式添加到对角线) ---
            diag = A.diagonal()  # 获取对角线视图
            diag += eta  # 添加化学硬度项

            # --- 5. 构建增广矩阵 (添加约束) ---
            A_aug = torch.zeros((num_atoms + 1, num_atoms + 1), dtype=dtype, device=device)
            A_aug[:num_atoms, :num_atoms] = A  # 主矩阵部分
            A_aug[-1, :num_atoms] = 1.0        # 约束行
            A_aug[:num_atoms, -1] = 1.0        # 约束列
            A_aug[-1, -1] = 0.0                # 右下角

            return A_aug

        N = len(inputs.batch)
        cell = inputs.cell.view(3, 3) 
        A_aug = build_qeq_matrix_A(eta, pos, cell, 6, 1.0e-5)
        b_aug = torch.zeros(N + 1, dtype=dtype, device=device)           # [N+1]
        b_aug[:N] = -chi           # [N]
        b_aug[N] = Q_total         # 标量

            # --- e. 求解线性方程组 A_aug * x_aug = b_aug ---
            #     x_aug = [q0, q1, ..., q(N-1), lambda]
            #     我们只需要前 N 个解，即原子电荷 q_eq
        # try:
            # 使用 LU 分解求解稠密线性系统 (推荐)
            # torch.linalg.solve 返回 x 使得 Ax = b
        x_aug_solution = torch.linalg.solve(A_aug, b_aug.unsqueeze(1)) # solve 需要列向量
        mol_q_eq = x_aug_solution[:N, 0] # 取解向量的前 N 个元素作为电荷 [N]
        lambda_sol = x_aug_solution[N, 0] # (通常不需要拉格朗日乘子)

        # except torch.linalg.LinAlgError as e:
        #     # 如果矩阵是奇异的 (不可逆)，则使用伪逆 (SVD) 来求解 (更鲁棒)
        #     print(f"Warning: Singular matrix for system. Using pseudo-inverse.")
        #     # 计算 A_aug 的伪逆
        #     A_pinv = torch.linalg.pinv(A_aug)
        #     # 用伪逆求解 x_aug = A_pinv * b_aug
        #     x_aug_solution = A_pinv @ b_aug.unsqueeze(1)
        #     mol_q_eq = x_aug_solution[:N, 0] # [N]

        return mol_q_eq, lambda_sol