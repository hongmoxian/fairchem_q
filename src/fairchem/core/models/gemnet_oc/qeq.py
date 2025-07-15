import torch
import torch.nn as nn
# import torch_geometric
from torch_runstats.scatter import scatter
# from typing import Optional
import numpy as np
from ase.data import chemical_symbols


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
                 coul_damping_beta=18.7, coul_damping_r0=2.2,
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
        :param coul_damping_r0: 库仑相互作用阻尼参数r0 (m)
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
        self.coul_damping_r0 = coul_damping_r0
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
        self.charge_mlp = MLP(256, self.charge_mlp_hidden_dims, 1).to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        self.charge_mlp.initialize_weights()
        
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
        max_iterations = 10000
        convergence_threshold = 0.01
        learning_rate = 0.1

        with torch.no_grad():
            charge = self.charge_mlp(node_feat).squeeze(-1)
            charge = self.charge_ub * torch.tanh(charge / self.charge_ub)
            
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

                charge = self.charge_ub * torch.tanh(charge / self.charge_ub)

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

    def predict_charge(self, node_feat, inputs):
        """
        预测原子电荷，在训练和推理时使用不同的策略
        """
        if self.training:
            if not self.charge_mlp_initialized:
                self.initialize_charge_mlp(node_feat, inputs)

        # 获取基础电荷预测
        charge = self.charge_mlp(node_feat).squeeze(-1).float()
        
        # 在训练模式下，添加元素特定的偏置
        # if self.training:
        #     for elem, indices in self._get_element_indices(inputs['atomic_numbers']).items():
        #         if elem in self.charge_biases:
        #             charge[indices] = charge[indices] + self.charge_biases[elem]
        
        # 应用tanh限制
        charge = self.charge_ub * torch.tanh(charge / self.charge_ub)
        
        # 调整总电荷以满足分子电荷约束
        sum_charge = scatter(charge, inputs.batch, dim=0,)
        natoms = scatter(torch.ones_like(inputs.batch, dtype=torch.float32), inputs.batch, dim=0,)
        diff_charge = (inputs['charge'] - sum_charge) / natoms
        pred_charge = charge + torch.gather(diff_charge, 0, inputs.batch)

        # 确保预测的电荷有梯度属性
        if not pred_charge.requires_grad:
            pred_charge.requires_grad_(True)

        return pred_charge

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
        :return: 库仑相互作用能量 (eV)
        """
        # 将距离从埃转换为米
        dij_meter = dij * ANGSTROM_TO_METER
        rij = torch.sqrt(torch.sum(torch.square(dij_meter), dim=-1))
        # 将 pred_charge 从元电荷 (e) 转换为库仑 (C)
        pred_charge_coulomb = pred_charge * ELECTRON_CHARGE
        prefactor_coul = self.ele_factor * pred_charge_coulomb[row] * pred_charge_coulomb[col] / rij
        damp_coul = torch.sigmoid(self.coul_damping_beta / self.coul_damping_r0 * (rij - self.coul_damping_r0))
        softplus_coul = nn.functional.softplus((rij - self.coul_damping_r0) / self.coul_damping_r0)
        r0 = self.coul_damping_r0
        fcoul = prefactor_coul * damp_coul * (rij / r0 / (1 + softplus_coul))**2

        ecoul = prefactor_coul * rij / r0 / (1 + softplus_coul)

        # compute erfc correction in inference, not available in trianing
        if g_ewald is not None:
            grij = g_ewald * rij
            expm2 = torch.exp(-grij * grij)
            t = 1.0 / (1.0 + self.ewald_p * grij)
            erfc = t * (self.ewald_a[0] + t * (self.ewald_a[1] + t * (self.ewald_a[2] + t * (self.ewald_a[3] + t * self.ewald_a[4])))) * expm2
            ecoul += prefactor_coul * (erfc - 1.0)
            fcoul += prefactor_coul * (erfc + self.ewald_f * grij * expm2 - 1.0)

        coul_fij = dij * (fcoul / rij / rij).unsqueeze(-1)
        coul_fij_ev_angstrom = coul_fij * ANGSTROM_TO_METER / EV_TO_JOULE
        ecoul_ev = ecoul / EV_TO_JOULE

        natoms = pred_charge.size(0)
        coul_energy = 0.5 * scatter(scatter(ecoul_ev, row, dim=0, dim_size=natoms), inputs.batch, dim=0, dim_size=inputs.batch.max() + 1)
        coul_force = scatter(coul_fij_ev_angstrom, row, dim=0, dim_size=natoms) - scatter(coul_fij_ev_angstrom, col, dim=0, dim_size=natoms)

        # 将能量从焦耳转换为电子伏特
        
        return coul_energy, coul_force
    
    def get_qeq_force(self, charge_energy, pred_charge, grad_outputs=None):
        # qeq_fij = torch.autograd.grad([charge_energy], [inputs['vector']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] 
        # if qeq_fij is None: # used for torch.jit.script
        #     qeq_fij_cast = torch.zeros(size=inputs['vector'].size(), device=inputs['vector'].device)
        # else:
        #     qeq_fij_cast = -1.0 * qeq_fij
        # qeq_force = scatter(qeq_fij_cast, row, dim=0, dim_size=natoms) - scatter(qeq_fij_cast, col, dim=0, dim_size=natoms)  # out of memory
        qeq_force = torch.autograd.grad(charge_energy, pred_charge, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
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
 

    def get_electronegativity_energy(self, node_feat, pred_charge, inputs, nmols):
        """
        计算电负性能量
        :param node_feat: 节点特征
        :param pred_charge: 预测的原子电荷 (e)
        :param inputs: 输入数据，包含mol_ids
        :return: 电负性能量 (eV)
        """
        


        pred_electronegativity = self.electronegativity_mlp(node_feat).squeeze(-1)
        pred_electronegativity_hardness = self.hardness_mlp(node_feat).squeeze(-1)
        # 将 pred_charge 从元电荷 (e) 转换为库仑 (C)
        pred_charge_coulomb = pred_charge * ELECTRON_CHARGE
        electronegativity_energy = pred_electronegativity ** 2 * pred_charge_coulomb + \
                                   pred_electronegativity_hardness ** 2 * pred_charge_coulomb * pred_charge_coulomb
        electronegativity_energy = scatter(electronegativity_energy, inputs.batch, dim=0, dim_size=nmols)
        # 将能量从焦耳转换为电子伏特
        electronegativity_energy_ev = electronegativity_energy / EV_TO_JOULE
        return electronegativity_energy_ev

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

    