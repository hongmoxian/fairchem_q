import torch
import torch.nn as nn
# import torch_geometric
from torch_runstats.scatter import scatter
# from typing import Optional
import numpy as np
from ase.data import chemical_symbols
from torch.autograd import grad
from torch.optim import LBFGS
import os
# import jax
# import jax.numpy as jnp
# from jaxopt import LBFGS


# 定义一些常量
ELECTRON_CHARGE = 1.602176634e-19  # 电子电荷 (C)
COULOMB_CONSTANT = 8.9875517923e9  # 库仑常数 (N m^2 C^-2)
ANGSTROM_TO_METER = 1e-10  # 埃到米的转换因子
EV_TO_JOULE = 1.602176634e-19  # 电子伏特到焦耳的转换因子




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

        self.charge_mlp_initialized = False
        # self.electronegativity_mlp = MLP(256, self.electronegativity_mlp_hidden_dims, 1).to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        # self.hardness_mlp = MLP(256, self.hardness_mlp_hidden_dims, 1).to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        self.charge_mlp = MLP(256, self.charge_mlp_hidden_dims, 1).to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        # self.charge_mlp.initialize_weights()
        
        # 添加可训练的电荷偏置参数
        self.name2eta = {
            "K":3.84,
            "C":10.13,
            "H":13.84,
            "O":13.36,
            "Ni":8.41,
            "N":11.76,
            "Au":6.27,
        }
        # self.name2eta = {
        #     "K": nn.Parameter(torch.tensor(self.name2eta_["K"])),
        #     "C": nn.Parameter(torch.tensor(self.name2eta_["C"])),
        #     "H": nn.Parameter(torch.tensor(self.name2eta_["H"])),
        #     "O": nn.Parameter(torch.tensor(self.name2eta_["O"])),
        #     "Ni": nn.Parameter(torch.tensor(self.name2eta_["Ni"])),
        #     "N": nn.Parameter(torch.tensor(self.name2eta_["N"])),
        # }
        # # self.name2eta = {
        # #     "K":  5,   # 很软，容易失电子
        # #     "C":  6.5,   # 中等硬度
        # #     "H":  7.2,   # 略软于C/O，但不应比金属还硬
        # #     "O":  8.0,   # 较硬
        # #     "Ni": 7.0,   # 过渡金属，中等偏硬（比K硬得多）
        # #     "N":  7.5,   # 比C硬，比O略软
        # # }
        
        self.name2chi = {
             "K":2.42,
            "C":5.34,
            "H":4.53,
            "O":8.74,
            "Ni":4.47,
            "N":6.9,
            "Au":4.44,
        }

        # self.name2chi = {
        #     "K": nn.Parameter(torch.tensor(self.name2chi_["K"])),
        #     "C": nn.Parameter(torch.tensor(self.name2chi_["C"])),
        #     "H": nn.Parameter(torch.tensor(self.name2chi_["H"])),
        #     "O": nn.Parameter(torch.tensor(self.name2chi_["O"])),
        #     "Ni": nn.Parameter(torch.tensor(self.name2chi_["Ni"])),
        #     "N": nn.Parameter(torch.tensor(self.name2chi_["N"])),
        # }
        # QEq parameters in eV (Mulliken definition)
        # self.name2chi = {
        #     "H": 7.176,
        #     "C": 6.262,
        #     "N": 7.232,
        #     "O": 7.540,
        #     "K": 2.421,
        #     "Ni": 4.398,
        # }

        # self.name2eta = {
        #     "H": 6.422,
        #     "C": 4.999,
        #     "N": 7.302,   # Note: A(N) is negative!
        #     "O": 6.079,
        #     "K": 1.920,
        #     "Ni": 3.242,
        # }
        self.initialized = False
        # if not self.initialized:
        self.pretrain = False
        # # 初始化网络参数
        #     self._initialize_weights()
        # if os.path.exists("/home/wuzhihong/dp/fairchem/fairchem/ceshi/clam/fairchem_q/k-co2/q-no-mean/charge-v3/charge-v4/checkpoints/2025-10-17-11-52-32/checkpoint.pt"):
        #     self.pretrain = True
        #     # self.qeq_module = torch.load(self.config["qeq_model_path"])
        #     model = torch.load("/home/wuzhihong/dp/fairchem/fairchem/ceshi/clam/fairchem_q/k-co2/q-no-mean/charge-v3/charge-v4/checkpoints/2025-10-17-11-52-32/checkpoint.pt", map_location=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        #     self.electronegativity_mlp.load_state_dict(model["chi"])
        #     self.hardness_mlp.load_state_dict(model["eta"])

        #     for param in self.electronegativity_mlp.parameters():
        #         param.requires_grad = False
        #     for param in self.hardness_mlp.parameters():
        #         param.requires_grad = False
        #     name2chi_params = model["name2chi_params"] # 假设这是您之前保存的参数字典
        #     for key, tensor_value in name2chi_params.items():
        #         if key in self.name2chi:
        #             with torch.no_grad():
        #                 self.name2chi[key].data.copy_(tensor_value)

        #     # 2. 【新增】遍历 name2chi 字典，冻结所有参数
        #     for param in self.name2chi.values():
        #         param.requires_grad = False
    
    def _initialize_weights(self, node_feat, atomic_numbers):
        """初始化网络参数，使初始输出 = 元素经验值"""
        # 初始化最后一层权重为 0
        # nn.init.zeros_(self.electronegativity_mlp.mlp[-1].weight)
        # nn.init.zeros_(self.hardness_mlp.mlp[-1].weight)
        
        # # 初始化最后一层偏置为 0
        # nn.init.zeros_(self.electronegativity_mlp.mlp[-1].bias)
        # nn.init.zeros_(self.hardness_mlp.mlp[-1].bias)
        
        # self.initialized = True

        # def forward(self, node_feat, atomic_numbers):
        # 预测 delta_chi 和 delta_eta
        delta_chi = self.electronegativity_mlp(node_feat).squeeze(-1)  # [N]
        # for name, param in self.electronegativity_mlp.named_parameters():
        #     grad_status = "✓ 可训练" if param.requires_grad else "✗ 已冻结"
        #     print(f"{name}: {grad_status}")
        # delta_eta = 0.8*torch.tanh(self.hardness_mlp(node_feat).squeeze(-1))           # [N]
        
        # 获取基础 chi 和 eta
        base_chi = torch.zeros_like(atomic_numbers, dtype=torch.float32)
        base_eta = torch.zeros_like(atomic_numbers, dtype=torch.float32)
        
        for i, atomic_num in enumerate(atomic_numbers):
            symbol = chemical_symbols[int(atomic_num)]
            if symbol in self.name2chi:
                base_chi[i] = self.name2chi[symbol]
            if symbol in self.name2eta:
                base_eta[i] = self.name2eta[symbol]
        
        # 最终输出
        chi =  base_chi + delta_chi
        eta = base_eta 
        eta = torch.clamp(eta, min=1.0)  # 保证 > 1.0 eV
        
        return chi, eta

    
    # def initialize_charge_mlp(self, node_feat, inputs):
    #     """安全初始化方法，通过修改网络参数实现"""
    #     # with torch.no_grad():
    #         # 1. 构建元素类型到索引的映射
        # elem_to_indices = defaultdict(list)
        # for idx, atomic_num in enumerate(inputs['atomic_numbers']):
        #     symbol = chemical_symbols[int(atomic_num)]
        #     elem_to_indices[symbol].append(idx)

    #     # 2. 初始化偏置项（推荐方案）
    #     for elem, indices in elem_to_indices.items():
    #         if elem in name2chi:
    #             # 计算当前batch的均值差
    #             current_chi = self.electronegativity_mlp(node_feat[indices]).mean()
    #             bias = self.chi_biases[elem] - current_chi
    #             # 直接修改最后一层偏置参数
    #             self.electronegativity_mlp.mlp[-1].bias.data[indices] += bias

    #         if elem in name2eta:
    #             current_eta = self.hardness_mlp(node_feat[indices]).mean()
    #             bias = self.eta_biases[elem] - current_eta
    #             self.hardness_mlp.mlp[-1].bias.data[indices] += bias


            # 更新charge_mlp的最后一层偏置，使其产生所需的初始分布

    def predict_charge(self, node_feat, inputs):
        """
        预测原子电荷，在训练和推理时使用不同的策略
        """
        # if self.training:
        #     if not self.charge_mlp_initialized:
        #         self.initialize_charge_mlp(node_feat, inputs)

        # 获取基础电荷预测
        charge = self.charge_mlp(node_feat).squeeze(-1).to(torch.float32)
        # charge = torch.zeros_like(inputs['atomic_numbers'], dtype=torch.float32)
        
        # # 在训练模式下，添加元素特定的偏置
        # for i, atomic_num in enumerate(inputs['atomic_numbers']):
        #     symbol = chemical_symbols[int(atomic_num)]
        #     if symbol in self.target_charge_dict:
        #         charge[i] = self.target_charge_dict[symbol]
        # charge = torch.tensor(inputs.bader, dtype=torch.float32).squeeze()
        # for elem, indices in self._get_element_indices(inputs['atomic_numbers']).items():
        #     if elem in self.target_charge_dict:
        #         charge[indices] = charge[indices] + self.target_charge_dict[elem]
        
        # 应用tanh限制
        # charge = self.charge_ub * torch.tanh(charge / self.charge_ub)
        
        # 计算电负性，用于加权
        # pred_electronegativity = self.electronegativity_mlp(node_feat).squeeze(-1)

        en_dict = {
            'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04,
            'O': 3.44, 'F': 3.98, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90,
            'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'K': 0.82, 'Ca': 1.00, 'Sc': 1.36,
            'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88,
            'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18,
            'Se': 2.48, 'Br': 2.96, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
            'Nb': 1.59, 'Mo': 2.16, 'Tc': 1.91, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20,
            'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.12,
            'I': 2.66, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13,
            'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.22,
            'Dy': 1.23, 'Ho': 1.24, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.26, 'Lu': 1.27,
            'Hf': 1.30, 'Ta': 1.50, 'W': 2.36, 'Re': 1.93, 'Os': 2.18, 'Ir': 2.20,
            'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02
        }   
        pred_electronegativity = torch.tensor([en_dict[chemical_symbols[int(i)]] for i in inputs['atomic_numbers']], device=inputs['atomic_numbers'].device)
        
        # # # 基于电负性计算权重 (逆比例，电负性高的原子权重小)
        epsilon = 1e-6  # 避免除零
        weights = 1.0 / (pred_electronegativity + epsilon)
        
        # # 计算每个分子的总charge和natoms
        sum_charge = scatter(charge, inputs.batch, dim=0)
        natoms = scatter(torch.ones_like(inputs.batch, dtype=torch.float32), inputs.batch, dim=0)
        
        # 计算每个分子的总权重
        sum_weights = scatter(weights, inputs.batch, dim=0)
        
        # 规范化权重 (per-molecule)
        normalized_weights = weights / torch.gather(sum_weights, 0, inputs.batch)
        
        # 计算每个分子的总diff
        total_diff = inputs['charge'] - sum_charge
        
        # 分配diff到每个原子
        diff_charge = normalized_weights * torch.gather(total_diff, 0, inputs.batch)
        
        pred_charge = charge + diff_charge

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

    def get_coulomb_energy_ewald(self, row, col, dij_vec, pred_charge, inputs=None, r_cutoff=6.0, accuracy=1e-5, if_grad=True):
        """
        使用α参数的Ewald求和计算（四项能量）
        使用埃（Å）作为基本单位，避免复数运算
        """
        # 初始化能量项
        real_space = 0.0
        reciprocal_space = 0.0
        self_energy = 0.0
        corr_energy = 0.0
        # if not if_grad:
        #     pos = inputs.pos.detach()
        #     dij_vec = dij_vec.detach()
        # else:
        pos = inputs.pos

        # # 单位转换常数 (使用Å单位)
        EV_ANGSTROM = 14.399645  # 转换因子: (e²/(4πε₀)) in eV·Å

        # # 直接使用Å单位
        r_cutoff_ang = r_cutoff
        device = dij_vec.device

        # # 计算距离 (从矢量计算)
        # dij_ang = torch.norm(dij_vec, dim=1)  # [num_edges]

        # # 计算α参数 (单位: 1/Å)
        alpha = torch.sqrt(-torch.log(torch.tensor(accuracy))) / r_cutoff_ang

        # # 晶胞处理 (单位: Å)
        cell = inputs.cell.view(3, 3)
        V = torch.abs(torch.det(cell))  # 体积 (Å³)
        # # c_vec_norm = torch.norm(cell[:, 2])
        # # ab = V / c_vec_norm
        # # 计算倒易基矢 (1/Å)，注意这里需要2π因子
        B = 2 * torch.pi * torch.linalg.inv(cell).T

        # # 1. 实空间项 (只考虑在截断半径内的原子对)
        # mask = (dij_ang > 1e-11) & (dij_ang < r_cutoff_ang)
        # dij_safe = dij_ang[mask] + 1e-11
        # erfc_term = torch.erfc(alpha * dij_safe) / dij_safe  # 使用α
        
        # # 只计算符合条件的原子对
        # row_masked = row[mask]
        # col_masked = col[mask]
        # real_space = 0.5 * torch.sum(
        #     pred_charge[row_masked] * pred_charge[col_masked] * erfc_term
        # )

        # # 2. 自能项 (使用α)
        self_energy = - (alpha / torch.sqrt(torch.tensor(torch.pi).to(torch.float32))) * torch.sum(pred_charge**2)

        # 3. 倒易空间项 (使用α参数和K²计算，避免复数运算)
        # 计算k网格
        # kmax = torch.ceil(2 * alpha * torch.sqrt(-torch.log(torch.tensor(accuracy)))).int()
# --- 倒易空间项 (slab: 2D periodic in xy) ---
        kmax = torch.ceil(2 * alpha * torch.sqrt(-torch.log(torch.tensor(accuracy, device=device)))).int()
        h_vals = torch.arange(-kmax, kmax+1, device=device)
        k_vals = torch.arange(-kmax, kmax+1, device=device)
        h_grid, k_grid = torch.meshgrid(h_vals, k_vals, indexing='ij')
        l_grid = torch.zeros_like(h_grid)  # 固定 l=0


        # 倒易基矢 (b1, b2 from B = 2π * inv(cell).T)
        b1 = B[:, 0]  # [3]
        b2 = B[:, 1]  # [3]
        b3 = B[:, 2]  # [3] <-- 使用完整的3D倒格矢
        # k = h*b1 + k*b2
        k_vecs = (h_grid.unsqueeze(-1) * b1 + 
              k_grid.unsqueeze(-1) * b2 +
              l_grid.unsqueeze(-1) * b3).reshape(-1, 3) # [n_k, 3]
        k_norms = torch.norm(k_vecs, dim=1)
        non_zero_mask = k_norms > 1e-11
        k_vecs = k_vecs[non_zero_mask]
        k_norms = k_norms[non_zero_mask]
        k_norms_sq = k_norms**2

        # 结构因子
        k_dot_r = torch.matmul(k_vecs, pos.T)  # [n_k, n_atoms]
        cos_kr = torch.cos(k_dot_r)
        sin_kr = torch.sin(k_dot_r)
        real_sum = torch.matmul(cos_kr, pred_charge)
        imag_sum = torch.matmul(sin_kr, pred_charge)
        S_k_sq = real_sum**2 + imag_sum**2

        # 能量
        exp_term = torch.exp(-k_norms_sq / (4 * alpha**2))
        coeff = (2 * torch.pi / V) * exp_term / k_norms_sq
        reciprocal_space = torch.sum(S_k_sq * coeff)

        # 4. 表面校正项 (使用α)
        Q_tot = torch.sum(pred_charge)

        def dipole_correction_full(q, positions, cell, Q_total=None):
            """
            根据图片公式计算完整的偶极修正能量
            
            Args:
                q: [N] 原子电荷
                positions: [N, 3] 原子坐标（Å）
                cell: [3, 3] 晶胞矩阵
                Q_total: 体系总电荷（可选，若为None则自动计算）
            
            Returns:
                E_dipole: 偶极修正能量（eV）
            """
            # 单位转换常数：e²/Å → eV
            EV_ANGSTROM = 14.399645
            
            # 计算晶胞参数
            V = torch.abs(torch.det(cell))  # 体积（Å³）
            L_z = torch.norm(cell.view(3, 3)[:, 2])       # z方向晶胞长度（Å）
            # ab = V / L_z
            
            # 提取z坐标（假设z方向是非周期方向）
            z = positions[:, 2]  # [N]
            z = z - L_z / 2
            
            # 计算各项
            M_z = torch.sum(q * z)                    # 偶极矩 M_z = Σ q_i z_i
            sum_qz2 = torch.sum(q * z**2)             # Σ q_i z_i²
            
            # 总电荷（若未提供则计算）
            if Q_total is None:
                Q_total = torch.sum(q)
            
            # 按图片公式计算
            term1 = M_z**2                            # M_z²
            term2 = Q_total**2 * sum_qz2              # Q_tot² Σ q_i z_i²  
            term3 = Q_total**2 * L_z**2 / 12          # Q_tot² L_z² / 12
            
            # 完整偶极修正
            E_dipole = (2 * torch.pi / V) * (term1 - term2 - term3)
            
            # 单位转换
            # E_dipole *= EV_ANGSTROM
            
            return E_dipole
        if abs(Q_tot) > 1e-11:
            # 根据 η² = 1/(2α²) 的关系，将公式中的 η 替换为 α
            # 原公式: - (π Q_tot²) / (2 η² V) = - (π Q_tot²) / (2 * (1/(2α²)) V) = - (π α² Q_tot²) / V
            # corr_energy = - (torch.pi * alpha**2 * Q_tot**2) / V
            corr_energy = dipole_correction_full(pred_charge, inputs.pos, inputs.cell, Q_tot)
            # M = torch.sum(pred_charge.unsqueeze(1) * inputs.pos, dim=0)  # [3]

    # 计算晶胞体积
            # V = torch.abs(torch.det(cell))

    # 表面校正项 (真空边界条件)
            # corr_energy = (2 * torch.pi / (3 * V)) * torch.sum(M**2)

        # 总能量 (单位: e²/Å)
        total_energy = real_space + reciprocal_space + self_energy + corr_energy

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
 

    def get_electronegativity_energy(self, pred_electronegativity, pred_electronegativity_hardness, pred_charge, inputs, nmols, if_grad=True):
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
        if not if_grad:
            pred_electronegativity = pred_electronegativity.detach()
            pred_electronegativity_hardness = pred_electronegativity_hardness.detach()
            # pred_charge = pred_charge.detach()
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
            inputs: 包含 pos, cell 等信息
            batch (torch.Tensor): [num_atoms] 每个原子属于哪个分子/体系的索引。
            Q_total (torch.Tensor): [num_systems] 每个体系的目标总电荷。
            edge_index (torch.Tensor, optional): [2, num_edges] 邻接表，指定哪些原子对之间有相互作用。
            edge_vec (torch.Tensor, optional): [num_edges, 3] 边矢量（考虑周期性）

        Returns:
            q_eq (torch.Tensor): [num_atoms] 求解得到的符合约束的原子电荷。
        """
        pos = inputs.pos
        device = pos.device
        dtype = pos.dtype
        N = len(batch)

        def build_qeq_matrix_A(eta, pos, cell, r_cutoff=6.0, accuracy=1e-5, edge_index=None, edge_vec=None):
            num_atoms = pos.shape[0]
            device = pos.device
            dtype = pos.dtype
            
            # 初始化 Coulomb 矩阵
            J = torch.zeros((num_atoms, num_atoms), dtype=dtype, device=device)

            # --- 1. 计算Ewald参数 ---
            alpha = torch.sqrt(-torch.log(torch.tensor(accuracy, device=device))) / r_cutoff
            V = torch.abs(torch.det(cell))
            B = 2 * torch.pi * torch.linalg.inv(cell).T.to(dtype)
            # c_vec_norm = torch.norm(cell[:, 2])
            # ab = V / c_vec_norm

            # --- 2. 实空间项 ---
            if edge_index is not None and edge_vec is not None:
                row, col = edge_index
                dij = torch.norm(edge_vec, dim=1)
                mask = (dij > 1e-10) & (dij < r_cutoff)
                dij_safe = dij[mask] + 1e-10
                erfc_term = torch.erfc(alpha * dij_safe) / dij_safe
                
                J[row[mask], col[mask]] = erfc_term
                J[col[mask], row[mask]] = erfc_term
            else:
                for i in range(num_atoms):
                    for j in range(i+1, num_atoms):
                        delta = pos[j] - pos[i]
                        frac_delta = torch.linalg.solve(cell.T, delta)
                        frac_delta = frac_delta - torch.round(frac_delta)
                        min_delta = frac_delta @ cell
                        dij = torch.norm(min_delta)
                        
                        if dij > 1e-10 and dij < r_cutoff:
                            dij_safe = dij + 1e-10
                            erfc_term = torch.erfc(alpha * dij_safe) / dij_safe
                            J[i, j] = erfc_term
                            J[j, i] = erfc_term

            # --- 3. 倒易空间项 ---
            kmax = torch.ceil(2 * alpha * torch.sqrt(-torch.log(torch.tensor(accuracy, device=device)))).int()

            b1 = B[:, 0]  # shape [3]
            b2 = B[:, 1]  # shape [3]
            b3 = B[:, 2]

            h_vals = torch.arange(-kmax, kmax + 1, device=device)
            k_vals = torch.arange(-kmax, kmax + 1, device=device)
            # l_vals = torch.arange(-kmax, kmax + 1, device=device)
            h_grid, k_grid = torch.meshgrid(h_vals, k_vals, indexing='ij')
            l_grid = torch.zeros_like(h_grid)  # 固定 l=0

            k_vecs = (h_grid.unsqueeze(-1) * b1 + 
              k_grid.unsqueeze(-1) * b2 +
              l_grid.unsqueeze(-1) * b3).reshape(-1, 3) # [n_k, 3]

            # 计算 |G|^2
            k_norms_sq = torch.sum(k_vecs**2, dim=1)  # [n_k]
            k_norms = torch.norm(k_vecs, dim=1)
            non_zero_mask = k_norms_sq > 1e-10
            k_vecs = k_vecs[non_zero_mask]
            k_norms_sq = k_norms_sq[non_zero_mask]
            k_norms = k_norms[non_zero_mask]

            if k_vecs.shape[0] == 0:
                # 没有非零倒格矢，跳过倒易空间
                recip_contribution = torch.zeros((num_atoms, num_atoms), dtype=dtype, device=device)
            else:
                # 计算 G · r_i for all G and atoms i
                k_dot_r = torch.matmul(k_vecs, pos.T)  # [n_k, n_atoms]

                # 权重因子: exp(-|G|^2 / (4α²)) / |G|^2
                exp_factor = torch.exp(-k_norms_sq / (4 * alpha**2)) / k_norms_sq  # [n_k]

                # cos(G·r_i) 和 sin(G·r_i)
                cos_kr = torch.cos(k_dot_r)  # [n_k, n_atoms]
                sin_kr = torch.sin(k_dot_r)  # [n_k, n_atoms]

                # 利用 cos(a - b) = cos a cos b + sin a sin b
                # 扩展维度以进行广播: [n_k, N, 1] * [n_k, 1, N] -> [n_k, N, N]
                cos_diff = (
                    cos_kr.unsqueeze(2) * cos_kr.unsqueeze(1) +
                    sin_kr.unsqueeze(2) * sin_kr.unsqueeze(1)
                )  # [n_k, num_atoms, num_atoms]

                # 加权求和 over G
                weighted_sum = (exp_factor.unsqueeze(1).unsqueeze(2) * cos_diff).sum(dim=0)  # [N, N]

                recip_contribution = (4 * torch.pi / V) * weighted_sum

            # 添加到 Coulomb 矩阵 J
            J += recip_contribution

            # --- 4. 自能修正 ---
            # 倒易空间已包含自相互作用，只需修正实空间的自能
            self_energy_correction = -2 * alpha / torch.sqrt(torch.tensor(torch.pi, device=device))
            # J.diagonal() += self_energy_correction

            # --- 5. 构建总矩阵 ---
            A = J + torch.diag(eta + self_energy_correction)  # 添加化学硬度

            # --- 6. 增广系统 ---
            A_aug = torch.zeros((num_atoms + 1, num_atoms + 1), dtype=dtype, device=device)
            A_aug[:num_atoms, :num_atoms] = A
            A_aug[:num_atoms, -1] = 1.0
            A_aug[-1, :num_atoms] = 1.0
            
            return A_aug

    # 获取晶胞
        cell = inputs.cell.view(3, 3)
        
        # 构建增广矩阵
        A_aug = build_qeq_matrix_A(eta, pos, cell, 6.0, 1.0e-5, edge_index, edge_vec)
        
        # 构建增广右端项
        b_aug = torch.zeros(N + 1, dtype=dtype, device=device)
        b_aug[:N] = -chi           # 电负性项（带负号！）
        b_aug[N] = Q_total         # 总电荷约束(体系电荷为负电荷，加个负号)

        # 求解线性方程组
        try:
            x_aug_solution = torch.linalg.solve(A_aug, b_aug.unsqueeze(1))
            mol_q_eq = x_aug_solution[:N, 0]
            lambda_sol = x_aug_solution[N, 0]
        except torch.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            A_pinv = torch.linalg.pinv(A_aug)
            x_aug_solution = A_pinv @ b_aug.unsqueeze(1)
            mol_q_eq = x_aug_solution[:N, 0]
            lambda_sol = x_aug_solution[N, 0]

        return mol_q_eq, lambda_sol

    # def compute_gradients(self, pred_charge, chi, J, inputs, dij_vec, row, col):
    #     """
    #     计算能量对电荷的梯度（自动微分实现）
    #     """
    #     pred_charge.requires_grad_(True)
    #     # E = total_energy(pred_charge, chi, J, positions, cell, dij_vec, row, col)
    #     E = self.get_coulomb_energy_ewald(row, col, dij_vec, pred_charge, inputs=None, r_cutoff=6.0, accuracy=1e-5) + self.get_electronegativity_energy(chi, J, pred_charge, inputs=inputs, nmols=inputs.batch.max() + 1)
    #     gradients = grad(E, pred_charge, create_graph=True)[0]
    #     return gradients
    def solve_qeq_linear_system_from_c(self, chi, eta, inputs, batch, Q_total, edge_index=None, edge_vec=None):
        """
        使用矩阵求逆法求解 QEq 方程，结合 C++ 代码逻辑构建矩阵
        
        Args:
            chi (torch.Tensor): [num_atoms] 预测的每个原子的电负性
            eta (torch.Tensor): [num_atoms] 预测的每个原子的化学硬度
            inputs: 包含 pos, cell 等信息
            batch (torch.Tensor): [num_atoms] 每个原子属于哪个分子/体系的索引
            Q_total (torch.Tensor): [num_systems] 每个体系的目标总电荷
            edge_index (torch.Tensor, optional): [2, num_edges] 邻接表
            edge_vec (torch.Tensor, optional): [num_edges, 3] 边矢量（考虑周期性）
            
        Returns:
            q_eq (torch.Tensor): [num_atoms] 求解得到的符合约束的原子电荷
            lambda_sol (torch.Tensor): 拉格朗日乘子
        """
        pos = inputs.pos
        device = pos.device
        dtype = pos.dtype
        N = len(batch)
        lambda_val = 1.2  # C++ 中的默认值
        k_const = 14.4    # 物理常数 k = 14.4 eV·Å (1/(4πε0))
        
        def build_qeq_matrix_A(eta, pos, cell, r_cutoff=6.0, accuracy=1e-5, 
                            edge_index=None, edge_vec=None):
            num_atoms = pos.shape[0]
            device = pos.device
            dtype = pos.dtype
            
            # 初始化 Coulomb 矩阵
            J = torch.zeros((num_atoms, num_atoms), dtype=dtype, device=device)
            
            # --- 1. 计算Ewald参数 ---
            alpha = torch.sqrt(-torch.log(torch.tensor(accuracy, device=device))) / r_cutoff
            V = torch.abs(torch.det(cell))
            B = 2 * torch.pi * torch.linalg.inv(cell).T.to(dtype)
            
            # --- 2. 实空间项 ---
            if edge_index is not None and edge_vec is not None:
                row, col = edge_index
                dij = torch.norm(edge_vec, dim=1)
                mask = (dij > 1e-10) & (dij < r_cutoff)
                dij_safe = dij[mask] + 1e-10
                
                # 添加轨道重叠项
                Jij = torch.sqrt(eta[row[mask]] * eta[col[mask]])
                a = Jij / k_const
                orbital_term = torch.exp(-(a**2 * dij_safe**2)) * (2*a - a**2*dij_safe - 1/dij_safe)
                
                erfc_term = torch.erfc(alpha * dij_safe) / dij_safe + orbital_term
                
                J[row[mask], col[mask]] = lambda_val * (k_const/2) * erfc_term
                J[col[mask], row[mask]] = lambda_val * (k_const/2) * erfc_term
            else:
                for i in range(num_atoms):
                    for j in range(i+1, num_atoms):
                        delta = pos[j] - pos[i]
                        frac_delta = torch.linalg.solve(cell.T, delta)
                        frac_delta = frac_delta - torch.round(frac_delta)
                        min_delta = frac_delta @ cell
                        dij = torch.norm(min_delta)
                        
                        if dij > 1e-10 and dij < r_cutoff:
                            dij_safe = dij + 1e-10
                            
                            # 添加轨道重叠项
                            Jij = torch.sqrt(eta[i] * eta[j])
                            a = Jij / k_const
                            orbital_term = torch.exp(-(a**2 * dij_safe**2)) * (2*a - a**2*dij_safe - 1/dij_safe)
                            
                            erfc_term = torch.erfc(alpha * dij_safe) / dij_safe + orbital_term
                            
                            J[i, j] = lambda_val * (k_const/2) * erfc_term
                            J[j, i] = lambda_val * (k_const/2) * erfc_term
            
            # --- 3. 倒易空间项 ---
            kmax = torch.ceil(2 * alpha * torch.sqrt(-torch.log(torch.tensor(accuracy, device=device)))).int()
            
            b1 = B[:, 0]
            b2 = B[:, 1]
            b3 = B[:, 2]
            
            h_vals = torch.arange(-kmax, kmax+1, device=device)
            k_vals = torch.arange(-kmax, kmax+1, device=device)
            h_grid, k_grid = torch.meshgrid(h_vals, k_vals, indexing='ij')
            l_grid = torch.zeros_like(h_grid)  # 固定 l=0
            
            k_vecs = (h_grid.unsqueeze(-1) * b1 + 
                    k_grid.unsqueeze(-1) * b2 +
                    l_grid.unsqueeze(-1) * b3).reshape(-1, 3)
            
            # 计算 |G|^2
            k_norms_sq = torch.sum(k_vecs**2, dim=1)
            non_zero_mask = k_norms_sq > 1e-10
            k_vecs = k_vecs[non_zero_mask]
            k_norms_sq = k_norms_sq[non_zero_mask]
            
            if k_vecs.shape[0] == 0:
                recip_contribution = torch.zeros((num_atoms, num_atoms), dtype=dtype, device=device)
            else:
                # 计算 G · r_i for all G and atoms i
                k_dot_r = torch.matmul(k_vecs, pos.T)
                
                # 权重因子: exp(-|G|^2 / (4α²)) / |G|^2
                exp_factor = torch.exp(-k_norms_sq / (4 * alpha**2)) / k_norms_sq
                
                # cos(G·r_i) 和 sin(G·r_i)
                cos_kr = torch.cos(k_dot_r)
                sin_kr = torch.sin(k_dot_r)
                
                # 利用 cos(a - b) = cos a cos b + sin a sin b
                cos_diff = (
                    cos_kr.unsqueeze(2) * cos_kr.unsqueeze(1) +
                    sin_kr.unsqueeze(2) * sin_kr.unsqueeze(1)
                )
                
                # 加权求和 over G
                weighted_sum = (exp_factor.unsqueeze(1).unsqueeze(2) * cos_diff).sum(dim=0)
                recip_contribution = lambda_val * (k_const/2) * (4 * torch.pi / V) * weighted_sum
            
            # 添加到 Coulomb 矩阵 J
            J += recip_contribution
            
            # --- 4. 自能修正 ---
            self_energy_correction = -2 * alpha / torch.sqrt(torch.tensor(torch.pi, device=device))
            for i in range(num_atoms):
                # 对角元添加化学硬度和自能修正
                J[i, i] = eta[i] + lambda_val * (k_const/2) * self_energy_correction
                
                # 添加实空间自能项（周期性边界）
                for u in range(-2, 3):  # mR=2
                    for v in range(-2, 3):
                        for w in range(-2, 3):
                            if u == 0 and v == 0 and w == 0:
                                continue
                            delta = torch.tensor([u, v, w], dtype=dtype, device=device) @ cell
                            dij = torch.norm(delta)
                            if dij > 1e-10 and dij < r_cutoff:
                                dij_safe = dij + 1e-10
                                erfc_term = torch.erfc(alpha * dij_safe) / dij_safe
                                J[i, i] += lambda_val * (k_const/2) * erfc_term
            
            return J
        
        # 获取晶胞
        cell = inputs.cell.view(3, 3)
        
        # 构建完整的 J 矩阵
        J_full = build_qeq_matrix_A(eta, pos, cell, 6.0, 1.0e-5, edge_index, edge_vec)
        
        # 按照原始代码格式构建增广矩阵
        # --- 5. 构建总矩阵 ---
        A = J_full  # 在 C++ 逻辑中，J_full 已经包含了化学硬度和自能修正
        
        # --- 6. 增广系统 ---
        A_aug = torch.zeros((N + 1, N + 1), dtype=dtype, device=device)
        A_aug[:N, :N] = A
        A_aug[:N, -1] = 1.0
        A_aug[-1, :N] = 1.0
        
        # 构建增广右端项
        b_aug = torch.zeros(N + 1, dtype=dtype, device=device)
        b_aug[:N] = -chi           # 电负性项（带负号！）
        b_aug[N] = Q_total         # 总电荷约束
        
        # 求解线性方程组
        try:
            x_aug_solution = torch.linalg.solve(A_aug, b_aug.unsqueeze(1))
            mol_q_eq = x_aug_solution[:N, 0]
            lambda_sol = x_aug_solution[N, 0]
        except torch.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            A_pinv = torch.linalg.pinv(A_aug)
            x_aug_solution = A_pinv @ b_aug.unsqueeze(1)
            mol_q_eq = x_aug_solution[:N, 0]
            lambda_sol = x_aug_solution[N, 0]
        
        return mol_q_eq, lambda_sol

    
    
    def solve_qeq(self, initial_charge, chi, J, inputs, dij_vec, row, col, Q_total, max_iter=100):
        
        """
        使用 L-BFGS 隐式求解 QEq 方程
        """
        device = inputs.pos.device
        q = torch.nn.Parameter(initial_charge.detach().clone()).to(device)
        Q_total = Q_total.to(device)
        # optimizer = LBFGS([q], lr=0.1, max_iter=max_iter)
        lam = torch.nn.Parameter(torch.zeros(1, device=device))
        optimizer = LBFGS([q, lam], lr=0.00001, max_iter=max_iter)
        # self.ewald_energy = self.get_coulomb_energy_ewald(row, col, dij_vec, q, inputs=inputs, r_cutoff=6.0, accuracy=1e-5)
        # self.electronegativity_energy = self.get_electronegativity_energy(chi, J, q, inputs=inputs, nmols=inputs.batch.max() + 1)
        # total_energy = self.ewald_energy + self.electronegativity_energy
        def closure():
            optimizer.zero_grad()
            for name, val in self.__dict__.items():
                if torch.is_tensor(val) and val.requires_grad:
                    raise RuntimeError(f"Found gradient tensor in self.{name}. Remove all 'self.xxx = tensor' in energy functions!")
        # ===================================
            # constraint = lam * (torch.sum(q) - Q_total)  # 约束项
            # loss = total_energy(q, chi, J, positions, cell, dij_vec, row, col)
            loss = self.get_coulomb_energy_ewald(row, col, dij_vec, q, inputs=inputs, r_cutoff=6.0, accuracy=1e-5, if_grad=False) + lam * (torch.sum(q) - Q_total) + self.get_electronegativity_energy(chi, J, q, inputs=inputs, nmols=inputs.batch.max() + 1, if_grad=False)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return q.detach(), lam.detach()

#     import torch
# from torch_geometric.utils import scatter

    # def solve_qeq_linear_system(self, chi, eta, inputs, batch, Q_total, edge_index=None, edge_vec=None, r_cutoff=6.0, accuracy=1e-5, use_dipole_corr=False):
    #     """
    #     使用线性方程组求解 QEq 电荷（支持多体系 batch）
        
    #     Args:
    #         chi: [num_atoms]
    #         eta: [num_atoms]
    #         inputs: 包含 pos ([N,3]), cell ([num_sys, 3, 3])
    #         batch: [num_atoms] 体系索引
    #         Q_total: [num_systems] 每个体系总电荷
    #         edge_index: [2, num_edges]
    #         edge_vec: [num_edges, 3] (考虑 PBC 的边矢量)
    #         use_dipole_corr: 是否使用偶极校正（slab 体系推荐 True）
    #     Returns:
    #         q_eq: [num_atoms], lambda_sol: [num_systems]
    #     """
    #     device = chi.device
    #     dtype = chi.dtype
    #     q_eq_list = []
    #     lambda_list = []

    #     # 按体系分割
    #     system_indices = torch.unique(batch)
    #     for sys_idx in system_indices:
    #         mask = (batch == sys_idx)
    #         sys_pos = inputs.pos[mask]
    #         sys_eta = eta[mask]
    #         sys_chi = chi[mask]
    #         sys_Q_total = Q_total[sys_idx]
    #         sys_cell = inputs.cell[sys_idx].view(3, 3)  # [3,3]

    #         n_atoms = sys_pos.shape[0]
    #         if n_atoms == 0:
    #             continue

    #         # 提取该体系的边
    #         if edge_index is not None and edge_vec is not None:
    #             edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    #             sys_edge_index = edge_index[:, edge_mask]
    #             sys_edge_vec = edge_vec[edge_mask]
    #             # 重映射原子索引到局部 [0, n_atoms)
    #             local_map = torch.zeros(mask.shape[0], dtype=torch.long, device=device)
    #             local_map[mask] = torch.arange(n_atoms, device=device)
    #             sys_edge_index = local_map[sys_edge_index]
    #         else:
    #             sys_edge_index = None
    #             sys_edge_vec = None

    #         # === 构建 Coulomb 矩阵 J (i≠j) ===
    #         J = torch.zeros((n_atoms, n_atoms), dtype=dtype, device=device)

    #         # Ewald 参数
    #         alpha = torch.sqrt(-torch.log(torch.tensor(accuracy, device=device))) / r_cutoff
    #         V = torch.abs(torch.det(sys_cell))
    #         B = 2 * torch.pi * torch.linalg.inv(sys_cell).T.to(dtype)

    #         # --- 实空间项 ---
    #         if sys_edge_index is not None:
    #             row, col = sys_edge_index
    #             dij = torch.norm(sys_edge_vec, dim=1)
    #             mask_r = (dij > 1e-10) & (dij < r_cutoff)
    #             if mask_r.any():
    #                 dij_safe = dij[mask_r] + 1e-10
    #                 erfc_term = torch.erfc(alpha * dij_safe) / dij_safe
    #                 J[row[mask_r], col[mask_r]] = erfc_term
    #                 J[col[mask_r], row[mask_r]] = erfc_term
    #         else:
    #             # 全连接（小体系）
    #             for i in range(n_atoms):
    #                 for j in range(i+1, n_atoms):
    #                     delta = sys_pos[j] - sys_pos[i]
    #                     frac_delta = torch.linalg.solve(sys_cell.T, delta)
    #                     frac_delta = frac_delta - torch.round(frac_delta)
    #                     min_delta = frac_delta @ sys_cell
    #                     dij = torch.norm(min_delta)
    #                     if dij > 1e-10 and dij < r_cutoff:
    #                         dij_safe = dij + 1e-10
    #                         erfc_term = torch.erfc(alpha * dij_safe) / dij_safe
    #                         J[i, j] = erfc_term
    #                         J[j, i] = erfc_term

    #         # --- 倒易空间项 (3D, 若为 slab 应改为 2D) ---
    #         kmax = torch.ceil(2 * alpha * torch.sqrt(-torch.log(torch.tensor(accuracy, device=device)))).int()
    #         k = torch.arange(-kmax, kmax + 1, device=device)
    #         kx, ky, kz = torch.meshgrid(k, k, 0, indexing='ij')
    #         k_grid = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3).to(dtype)
    #         k_vecs = k_grid @ B.T
    #         k_norms = torch.norm(k_vecs, dim=1)
    #         non_zero_mask = k_norms > 1e-10
    #         k_vecs = k_vecs[non_zero_mask]
    #         k_norms = k_norms[non_zero_mask]

    #         # 全原子对计算倒易空间（不能用 edge_index！）
    #         for i in range(n_atoms):
    #             for j in range(n_atoms):
    #                 if i == j:
    #                     continue
    #                 delta = sys_pos[j] - sys_pos[i]
    #                 frac_delta = torch.linalg.solve(sys_cell.T, delta)
    #                 frac_delta = frac_delta - torch.round(frac_delta)
    #                 min_delta = frac_delta @ sys_cell
    #                 k_dot_r_ij = torch.matmul(k_vecs, min_delta)
    #                 cos_kr_ij = torch.cos(k_dot_r_ij)
    #                 recip_term = (2 * torch.pi / V) * torch.sum(
    #                     torch.exp(-k_norms**2 / (4 * alpha**2)) / k_norms**2 * cos_kr_ij
    #                 )
    #                 J[i, j] += recip_term

    #         # --- 构建总矩阵 A ---
    #         # 自能项合并到 eta
    #         self_interaction = alpha / torch.sqrt(torch.tensor(torch.pi, device=device))
    #         eta_eff = sys_eta + self_interaction

    #         A = J + torch.diag(eta_eff)

    #         # --- 偶极校正项（slab 体系）---
    #         if use_dipole_corr:
    #             # C_ij = (2π / 3V) * (r_i · r_j)
    #             C = (2 * torch.pi / (3 * V)) * (sys_pos @ sys_pos.T)
    #             A = A + C

    #         # --- 增广矩阵 ---
    #         A_aug = torch.zeros((n_atoms + 1, n_atoms + 1), dtype=dtype, device=device)
    #         A_aug[:n_atoms, :n_atoms] = A
    #         A_aug[:n_atoms, -1] = -1.0   # ← 关键：-1
    #         A_aug[-1, :n_atoms] = 1.0
    #         A_aug[-1, -1] = 0.0

    #         # --- 右端项 ---
    #         b_aug = torch.zeros(n_atoms + 1, dtype=dtype, device=device)
    #         b_aug[:n_atoms] = -sys_chi
    #         b_aug[n_atoms] = sys_Q_total

    #         # --- 求解 ---
    #         try:
    #             x = torch.linalg.solve(A_aug, b_aug.unsqueeze(1))
    #         except torch.linalg.LinAlgError:
    #             x = torch.linalg.pinv(A_aug) @ b_aug.unsqueeze(1)

    #         q_eq_list.append(x[:n_atoms, 0])
    #         lambda_list.append(x[n_atoms, 0])

    #     q_eq = torch.cat(q_eq_list, dim=0) if q_eq_list else torch.empty(0, device=device)
    #     lambda_sol = torch.cat(lambda_list, dim=0) if lambda_list else torch.empty(0, device=device)
    #     return q_eq, lambda_sol