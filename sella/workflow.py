import os
import shutil
import time
import subprocess
import argparse
import pickle
import json
from turtle import st
import numpy as np
import torch
import lmdb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from ase import io, Atoms
from ase.constraints import FixAtoms
from ase.neb import NEB
from sella import Sella

# 引入 Fairchem/OCP 相关库
from fairchem.core import OCPCalculator
from fairchem.core.preprocessing import AtomsToGraphs
# from glob import glob
from ase.io import write
from ase.neb import NEB
from ase.optimize import BFGS, FIRE

# ==========================================
# 数据结构定义
# ==========================================
class Stage(Enum):
    INTERPOLATION = "interpolation"
    VASP_SAMPLING = "vasp_sampling"
    LMDB_CREATION = "lmdb_creation"
    FINE_TUNING = "fine_tuning"
    SELLA_SEARCH = "sella_search"
    COMPLETED = "completed"

@dataclass
class CheckpointData:
    """保存断点信息的数据结构"""
    completed_stages: List[str]
    current_stage: Optional[str] = None
    images: Optional[List[str]] = None
    task_info: Optional[List[Dict]] = None
    job_ids: Optional[List[str]] = None
    ts_guess_path: Optional[str] = None
    model_path: Optional[str] = None
    timestamp: float = 0.0

# ==========================================
# 断点管理器
# ==========================================
class CheckpointManager:
    def __init__(self, checkpoint_file="checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.data = None
        self._load()
    
    def _load(self):
        """加载断点信息"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_dict = json.load(f)
                self.data = CheckpointData(**checkpoint_dict)
            print(f"✅ 从断点恢复：已完成阶段={self.data.completed_stages}")
        else:
            self.data = CheckpointData(
                completed_stages=[],
                current_stage=Stage.INTERPOLATION.value,
                timestamp=time.time()
            )
    
    def save(self, stage: Stage, **kwargs):
        """保存断点信息"""
        self.data.current_stage = stage.value
        self.data.timestamp = time.time()
        
        for key, value in kwargs.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(asdict(self.data), f, indent=2)
        print(f"💾 保存断点：{stage.value}")
    
    def mark_completed(self, stage: Stage):
        """标记当前阶段完成"""
        if stage.value not in self.data.completed_stages:
            self.data.completed_stages.append(stage.value)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(asdict(self.data), f, indent=2)
        print(f"✅ 标记阶段完成：{stage.value}")
    
    def is_completed(self, stage: Stage) -> bool:
        """检查某个阶段是否已完成"""
        return stage.value in self.data.completed_stages
    
    def clear(self):
        """清除断点"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        self.data = None

# ==========================================
# 工具函数 (保持原样)
# ==========================================
def get_tags(ase_data: Atoms) -> np.ndarray:
    """用户自定义Tag函数"""
    structure = ase_data
    exclude_elements = ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'I', 'F']
    tags = np.ones(len(structure)) * 2
    
    non_excluded_indices = []
    non_excluded_z_coords = []
    
    for idx, atom in enumerate(structure):
        if atom.symbol not in exclude_elements:
            non_excluded_indices.append(idx)
            non_excluded_z_coords.append(atom.position[2])
    
    if len(non_excluded_z_coords) < 2:
        return tags
    
    z_min = np.min(non_excluded_z_coords)
    z_max = np.max(non_excluded_z_coords)
    
    for idx, z in zip(non_excluded_indices, non_excluded_z_coords):
        if abs(z - z_min) <= 0.5 or abs(z - z_max) <= 0.5:
            tags[idx] = 1
        else:
            tags[idx] = 0
    
    return tags

def read_force_constants(filename: str = 'FORCE_CONSTANTS') -> np.ndarray:
    """读取FORCE_CONSTANTS文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0].split()[0])
    fc_matrix = np.zeros((num_atoms * 3, num_atoms * 3))
    
    idx = 1
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        
        i, j = map(int, line.split())
        i -= 1
        j -= 1
        
        fc_submatrix = np.zeros((3, 3))
        for k in range(3):
            fc_submatrix[k] = list(map(float, lines[idx + 1 + k].split()))
        
        fc_matrix[3*i:3*i+3, 3*j:3*j+3] = fc_submatrix
        idx += 4
    
    return fc_matrix

# ==========================================
# VASP 任务管理器 (修复版)
# ==========================================
class VaspManager:
    def __init__(self, base_dir: str, utils_dir: str, ts_guess_idx: int = 3):
        self.base_dir = Path(base_dir).resolve()
        self.utils_dir = Path(utils_dir).resolve()
        self.ts_guess_idx = ts_guess_idx
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.utils_dir.exists():
            raise FileNotFoundError(f"Utils目录不存在: {self.utils_dir}")
    
    def prepare_tasks(self, images: List[Atoms]) -> Tuple[List[str], List[Dict]]:
        """
        准备VASP任务
        
        Args:
            images: 所有插值得到的结构列表
        Returns:
            job_ids: 所有任务的作业ID
            task_info: 任务信息列表
        """
        job_ids = []
        task_info = []
        
        for i, atoms in enumerate(images):
            is_middle = True
            
            image_dir = self.base_dir / f"image_{i:02d}"
            image_dir.mkdir(exist_ok=True)
            
            # 保存初始结构
            io.write(image_dir / "POSCAR_initial", atoms)
            
            # SP计算任务
            sp_dir = image_dir / "sp"
            sp_dir.mkdir(exist_ok=True)
            self._setup_vasp_folder(sp_dir, atoms, 'sp')
            job_id_sp = self._submit_job(sp_dir)
            job_ids.append(job_id_sp)
            
            task_dict = {
                'id': i,
                'image_dir': str(image_dir),
                'sp_path': str(sp_dir),
                'type': 'mixed' if is_middle else 'sp_only'
            }
            
            if is_middle:
                # Hessian计算任务
                hess_dir = image_dir / "hess"
                hess_dir.mkdir(exist_ok=True)
                self._setup_vasp_folder(hess_dir, atoms, 'hess')
                job_id_hess = self._submit_job(hess_dir)
                job_ids.append(job_id_hess)
                task_dict['hess_path'] = str(hess_dir)
                print(f"📤 提交Hessian计算: Image {i} -> Job {job_id_hess}")
            
            task_info.append(task_dict)
            print(f"📤 提交SP计算: Image {i} -> Job {job_id_sp}")
        
        return job_ids, task_info
    
    def _setup_vasp_folder(self, folder: Path, atoms: Atoms, mode: str):
        """设置VASP计算文件夹"""
        io.write(folder / "POSCAR", atoms)
        
        # 复制必要文件
        for f in ['POTCAR', 'KPOINTS', 'sub.vasp']:
            src = self.utils_dir / f
            if src.exists():
                shutil.copy(src, folder / f)
            else:
                print(f"⚠️  警告: 文件 {f} 不存在于 {self.utils_dir}")
        
        # 处理INCAR
        incar_template = self.utils_dir / "INCAR"
        if incar_template.exists():
            with open(incar_template, 'r') as f:
                base_incar = f.read()
        else:
            base_incar = "SYSTEM = NEB\nENCUT = 500\nISMEAR = 0\nSIGMA = 0.05\n"
        
        extra_flags = ""
        if mode == 'hess':
            extra_flags = "\nIBRION = 5\nNFREE = 1\nEDIFF = 1E-5\n"
        else:
            extra_flags = "\nIBRION = 2\nNSW=1\nEDIFF = 1E-5\n"
        
        with open(folder / "INCAR", 'w') as f:
            f.write(base_incar + extra_flags)
    
    def _submit_job(self, folder: Path) -> str:
        """提交作业到集群"""
        cwd = Path.cwd()
        os.chdir(folder)
        
        try:
            result = subprocess.run(
                ['sbatch', 'sub.vasp'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
            else:
                print(f"❌ 提交作业失败: {result.stderr}")
                job_id = f"failed_{int(time.time())}"
        except Exception as e:
            print(f"❌ 提交作业异常: {e}")
            job_id = f"error_{int(time.time())}"
        finally:
            os.chdir(cwd)
        
        return job_id
    
    def wait_all(self, job_ids: List[str], poll_interval: int = 30):
        """等待所有作业完成"""
        if not job_ids:
            return
        
        print(f"⏳ 等待 {len(job_ids)} 个作业完成...")
        
        while True:
            try:
                result = subprocess.run(
                    ['squeue', '--job', ','.join(job_ids)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                running_jobs = 0
                for line in result.stdout.strip().split('\n')[1:]:
                    if line.strip():
                        running_jobs += 1
                
                if running_jobs == 0:
                    break
                
                print(f"📊 仍在运行: {running_jobs}/{len(job_ids)} 个作业")
                time.sleep(poll_interval)
                
            except subprocess.CalledProcessError as e:
                print(f"❌ 检查作业状态失败: {e}")
                time.sleep(poll_interval)
        
        print("✅ 所有作业已完成")

# ==========================================
# LMDB 创建器
# ==========================================
class LMDBBuilder:
    def __init__(self, output_path: str = "dataset/train.lmdb"):
        self.output_path = Path(output_path)
        self.a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6,
            r_energy=True,
            r_forces=True,
            r_fixed=True,
        )
    
    def build(self, task_info: List[Dict]):
        """创建 LMDB 数据库"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🏗️  创建 LMDB 数据库：{self.output_path}")
        
        db = lmdb.open(
            str(self.output_path),
            map_size=1099511627776,  # 1TB
            subdir=False,
            meminit=False,
            map_async=True,
        )
        
        txn = db.begin(write=True)
        count = 0
        
        cwd = Path.cwd().resolve()
        
        for info in task_info:
            try:
                sp_path = Path(info['sp_path'])
                sp_outcar = sp_path / "OUTCAR"
                
                if not sp_outcar.exists():
                    print(f"⚠️  SP 结果不存在：{sp_outcar}")
                    continue
                
                atoms = io.read(str(sp_outcar), format='vasp-out')
                custom_tags = get_tags(atoms)
                atoms.set_tags(custom_tags)
                
                data_list = self.a2g.convert_all([atoms], disable_tqdm=True)
                data = data_list[0]
                
                n_atoms = len(atoms)
                hessian_matrix = None
                
                if info['type'] == 'mixed':
                    hess_path = Path(info.get('hess_path', ''))
                    fc_file = hess_path / "FORCE_CONSTANTS"

                    try:
                        os.chdir(str(hess_path))
                        result = subprocess.run(['phonopy', '--fc', 'vasprun.xml'], capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"⚠️  phonopy 执行失败：{result.stderr}")
                    finally:
                        os.chdir(str(cwd))
                    
                    if fc_file.exists():
                        hessian_matrix = read_force_constants(str(fc_file))

                if hessian_matrix is not None:
                    evals, _ = np.linalg.eigh(hessian_matrix)
                    data.hessian = torch.tensor(hessian_matrix, dtype=torch.float32)
                    data.hessian_mask = torch.tensor(1.0, dtype=torch.float32)
                    data.eigvals = torch.tensor(evals, dtype=torch.float32)
                else:
                    data.hessian = torch.zeros((3*n_atoms, 3*n_atoms), dtype=torch.float32)
                    data.hessian_mask = torch.tensor(0.0, dtype=torch.float32)
                    data.eigvals = torch.zeros(3*n_atoms, dtype=torch.float32)
                
                data.sid = torch.LongTensor([0])
                data.fid = torch.LongTensor([info['id']])
                
                txn.put(f"{count}".encode("ascii"), pickle.dumps(data, protocol=-1))
                count += 1
                
                print(f"✅ 处理 Image {info['id']}: Hessian={'Yes' if hessian_matrix is not None else 'No'}")
                
            except Exception as e:
                print(f"❌ 处理 Image {info['id']} 失败：{e}")
                import traceback
                traceback.print_exc()
        
        txn.put(b"length", pickle.dumps(count, protocol=-1))
        txn.commit()
        db.sync()
        db.close()
        
        print(f"✅ LMDB 创建完成：{count} 条记录")

# ==========================================
# 能垒计算
# ==========================================
def calculate_barrier_heights(initial_atoms: Atoms, ts_atoms: Atoms, final_atoms: Atoms) -> Dict:
    """计算能垒高度"""
    E_initial = initial_atoms.get_potential_energy()
    E_ts = ts_atoms.get_potential_energy()
    E_final = final_atoms.get_potential_energy()
    
    barriers = {
        'forward_barrier': E_ts - E_initial,
        'reverse_barrier': E_ts - E_final,
        'reaction_energy': E_final - E_initial,
        'activation_energy': max(E_ts - E_initial, E_ts - E_final)
    }
    
    print("\n" + "="*60)
    print("能垒分析")
    print("="*60)
    print(f"初始态能量: {E_initial:.4f} eV")
    print(f"过渡态能量: {E_ts:.4f} eV")
    print(f"终态能量:  {E_final:.4f} eV")
    print(f"正向能垒:  {barriers['forward_barrier']:.4f} eV")
    print(f"逆向能垒:  {barriers['reverse_barrier']:.4f} eV")
    print(f"反应能量:  {barriers['reaction_energy']:.4f} eV")
    print(f"活化能:    {barriers['activation_energy']:.4f} eV")
    
    return barriers

# ==========================================
# 主工作流
# ==========================================
class TSOptimizationWorkflow:
    def __init__(self, args):
        self.args = args
        self.checkpoint = CheckpointManager()
        self.vasp_manager = None
        self.images = []
        self.task_info = []
        self.job_ids = []
        
    def run(self):
        """运行完整工作流"""
        print("🚀 开始过渡态优化工作流")
        print(f"📊 已完成阶段：{self.checkpoint.data.completed_stages}")
        
        # 恢复必要的数据
        self._restore_data()
        
        stages = [
            Stage.INTERPOLATION,
            Stage.VASP_SAMPLING,
            Stage.LMDB_CREATION,
            Stage.FINE_TUNING,
            Stage.SELLA_SEARCH
        ]
        
        for stage in stages:
            if self.checkpoint.is_completed(stage):
                print(f"⏭️  跳过已完成阶段：{stage.value}")
                continue
            
            self._execute_stage(stage)
        
        print("✅ 工作流完成!")
        # self.checkpoint.clear()
    
    def _restore_data(self):
        """从断点恢复必要数据"""
        # 恢复 images
        if self.checkpoint.data.images:
            print(f"📁 从断点恢复 {len(self.checkpoint.data.images)} 个结构")
            self.images = []
            for img_file in self.checkpoint.data.images:
                if Path(img_file).exists():
                    self.images.append(io.read(img_file))
                else:
                    print(f"⚠️  警告：结构文件 {img_file} 不存在")
        
        # 恢复任务信息
        if self.checkpoint.data.task_info:
            self.task_info = self.checkpoint.data.task_info
            print(f"📊 恢复 {len(self.task_info)} 个任务信息")
        
        # 恢复作业ID
        if self.checkpoint.data.job_ids:
            self.job_ids = self.checkpoint.data.job_ids
    
    def _execute_stage(self, stage: Stage):
        """执行指定阶段的任务"""
        print(f"\n{'='*60}")
        print(f"执行阶段：{stage.value}")
        print('='*60)
        
        if stage == Stage.INTERPOLATION:
            self._run_interpolation()
        elif stage == Stage.VASP_SAMPLING:
            self._run_vasp_sampling()
        elif stage == Stage.LMDB_CREATION:
            self._run_lmdb_creation()
        elif stage == Stage.FINE_TUNING:
            self._run_fine_tuning()
        elif stage == Stage.SELLA_SEARCH:
            self._run_sella_search()
        
        self.checkpoint.mark_completed(stage)
    
    def _run_interpolation(self):
        """阶段 1: 插值"""
        print(">>> 阶段 1: 结构插值")
        
        ini = io.read(self.args.is_poscar)
        fin = io.read(self.args.fs_poscar)
        
        n_images = self.args.n_images
        self.images = [ini.copy()]
        for i in range(1, n_images - 1):
            self.images.append(ini.copy())
        self.images.append(fin.copy())
        
        neb = NEB(self.images)
        neb.interpolate(mic=True,method="idpp")
        
        # 保存插值结果
        image_files = []
        for i, atoms in enumerate(self.images):
            filename = f"image_{i:02d}.vasp"
            io.write(filename, atoms)
            image_files.append(filename)
        
        self.checkpoint.save(
            Stage.INTERPOLATION, 
            images=image_files,
            task_info=[],  # 清空之前的任务信息
            job_ids=[]     # 清空之前的作业ID
        )
    
    def _run_vasp_sampling(self):
        """阶段 2: VASP 采样"""
        print(">>> 阶段 2: VASP 采样")
        
        # 如果是从断点恢复，确保 images 已加载
        if not self.images and self.checkpoint.data.images:
            self.images = []
            for img_file in self.checkpoint.data.images:
                if Path(img_file).exists():
                    self.images.append(io.read(img_file))
        
        if not self.images:
            raise ValueError("没有可用的插值结构")
        
        self.vasp_manager = VaspManager(
            base_dir="vasp_work",
            utils_dir=self.args.utils_dir,
            ts_guess_idx=self.args.ts_guess_idx
        )
        
        self.job_ids, self.task_info = self.vasp_manager.prepare_tasks(self.images[self.args.ts_guess_idx-1:self.args.ts_guess_idx+2])  #00 - 06 / 2:5
        self.vasp_manager.wait_all(self.job_ids)
        
        self.checkpoint.save(
            Stage.VASP_SAMPLING, 
            task_info=self.task_info, 
            job_ids=self.job_ids
        )
    
    def _run_lmdb_creation(self):
        """阶段 3: LMDB 创建"""
        print(">>> 阶段 3: LMDB 创建")
        
        # 优先使用恢复的 task_info
        if not self.task_info and self.checkpoint.data.task_info:
            self.task_info = self.checkpoint.data.task_info
        
        if not self.task_info:
            raise ValueError("没有找到任务信息")
        
        # 验证 VASP 结果是否存在
        for info in self.task_info:
            sp_path = Path(info['sp_path'])
            if not (sp_path / "OUTCAR").exists():
                print(f"⚠️  警告：SP 结果不存在 {sp_path}/OUTCAR")
        
        lmdb_path = "dataset/train.lmdb"
        builder = LMDBBuilder(lmdb_path)
        builder.build(self.task_info)
    
    def _run_fine_tuning(self):
        """阶段 4: 微调"""
        print(">>> 阶段 4: 模型微调")
        
        # 检查 LMDB 是否存在
        lmdb_path = "dataset/train.lmdb"
        if not Path(lmdb_path).exists():
            raise FileNotFoundError(f"LMDB 文件不存在：{lmdb_path}")
        
        finetune_dir = Path("finetune_output").resolve()
        finetune_dir.mkdir(exist_ok=True)
        
        cwd = Path.cwd().resolve()
        os.chdir(str(finetune_dir))
        os.system('cp /home/wuzhihong/dp/fairchem/fairchem/ceshi/clam/hessian/finetune-hessian.yml .')
        os.system('cp /home/wuzhihong/dp/fairchem/fairchem/ceshi/clam/hessian/main.py  .')
        
        try:
            # 修改配置文件中的数据集路径
            
            cmd = f"""
                python main.py \
                    --mode train \
                    --config-yml finetune-hessian.yml \
                    --checkpoint {self.args.pretrained_ckpt} \
                    --amp \
                    --print-every 2 > output.log 2>&1
            """
            
            print(f"执行命令：{cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 微调失败：{result.stderr}")
                raise RuntimeError("微调失败")
            
            # 寻找最佳模型
            checkpoint_files = list(finetune_dir.glob("checkpoints/*/best_checkpoint.pt"))
            if not checkpoint_files:
                # 尝试其他可能的路径
                checkpoint_files = list(finetune_dir.glob("**/best_checkpoint.pt"))
            
            if not checkpoint_files:
                raise FileNotFoundError("未找到微调后的模型")
            
            model_path = checkpoint_files[0]
            print(f"✅ 找到模型：{model_path}")
            
        finally:
            os.chdir(str(cwd))
        
        self.checkpoint.save(Stage.FINE_TUNING, model_path=str(model_path))
    
    def neb_transition_state_search(self, initial_atoms, final_atoms, 
                               path, n_images=6, fmax=0.05, 
                               climb=True, method='aseneb'):
        """
        使用NEB方法寻找过渡态
        
        Args:
            initial_atoms: 初始结构 (ASE Atoms对象)
            final_atoms: 最终结构 (ASE Atoms对象)
            calculator: 计算器 (如VASP, GPAW, EMT等)
            n_images: 中间图像数量
            fmax: 收敛阈值 (eV/Å)
            climb: 是否使用CI-NEB (Climbing Image NEB)
            method: NEB方法 ('aseneb', 'improvedtangent', 'eb')
            
        Returns:
            neb_path: 包含所有图像的列表
            highest_image_index: 能量最高图像的索引
        """
        
        print("="*60)
        print("开始NEB过渡态搜索")
        print("="*60)
        
        # 设置计算器
        # initial_atoms.set_calculator(calculator)
        # final_atoms.set_calculator(calculator)
        
        # 创建初始NEB路径 (线性插值)
        images = [initial_atoms]
        images += [initial_atoms.copy() for _ in range(n_images - 2)]
        images += [final_atoms]
        
        # 设置插值
        neb = NEB(images, climb=False, method=method, k=1.5)
        
        # 设置计算器
        for image in images:
            image.set_calculator(OCPCalculator(checkpoint_path=path, cpu=not torch.cuda.is_available(), seed=123))
        
        # 插值得到初始路径
        neb.interpolate(mic=True,)
        
        print(f"初始路径创建完成，共{len(images)}个图像")
        print(f"初始能量: {initial_atoms.get_potential_energy():.4f} eV")
        print(f"最终能量: {final_atoms.get_potential_energy():.4f} eV")
        
        # 优化NEB路径
        print("\n开始NEB优化...")
        optimizer = FIRE(neb, trajectory='neb_path.traj', logfile='neb_opt.log')

        
    
        # optimizer.atoms
        optimizer.run(fmax=0.45, steps=100)

        neb.climb = True
        optimizer.run(fmax=0.1, steps=200)
        # 分析NEB结果
        
        # 获取能量最高点
        energies = [image.get_potential_energy() for image in images]
        highest_index = np.argmax(energies)
        barrier_height = energies[highest_index] - energies[0]
        
        print("\n" + "="*60)
        print("NEB优化完成")
        print(f"能量最高点索引: {highest_index}")
        print(f"能垒高度: {barrier_height:.4f} eV")
        print(f"能量最高点坐标:")
        
        # 保存NEB路径
        write("neb_final_path.vasp", images[highest_index])
        
        return images[highest_index]
    def _run_sella_search(self):
        """阶段 5: Sella 搜索"""
        print(">>> 阶段 5: Sella 过渡态搜索")
        
        import time 
        start_time = time.time()
        model_path = self.checkpoint.data.model_path
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        
        calc = OCPCalculator(
            checkpoint_path=model_path,
            cpu=not torch.cuda.is_available(),
            seed=123
        )
        self.images[0].set_tags(get_tags(self.images[0]))
        self.images[-1].set_tags(get_tags(self.images[-1]))


        # ts_guess = self.neb_transition_state_search(
        #     self.images[0],
        #     self.images[-1],
        #     model_path,
        #     n_images=7,
        #     fmax=0.05,
        #     climb=True,
        #     method='aseneb'
        # )

        def get_high_energy_image(images):
            for image in images:
                image.set_tags(get_tags(image))
                image.set_calculator(calc)
            energies = [image.get_potential_energy() for image in images]
            return images[np.argmax(energies)]
        
        ts_guess = get_high_energy_image(self.images).copy()
        ts_guess.set_tags(get_tags(ts_guess))
        ts_guess.calc = calc

        # 固定底层/体相原子，减少 slab 整体重排对 TS 搜索的干扰。
        # fixed_indices = np.where(np.asarray(ts_guess.get_tags()) == 0)[0]
        # if len(fixed_indices) > 0:
        #     ts_guess.set_constraint(FixAtoms(indices=fixed_indices.tolist()))
        #     print(f"🔒 固定底层原子数: {len(fixed_indices)}")
        # else:
        #     print("⚠️ 未识别到底层原子，Sella 将在无固定约束下运行")

        
        def get_hessian(atoms):
            return calc.get_property('hessian', atoms)

        print("🔍 开始过渡态预搜索（使用 Hessian）...")
        pre_dyn = Sella(
            ts_guess,
            trajectory='ts_pre.traj',
            logfile='ts_pre.log',
            hessian_function=get_hessian,
        )
        pre_dyn.run(fmax=0.1, steps=200)

        # ts_candidate = pre_dyn.atoms.copy()
        # ts_candidate.calc = calc
        # # if len(fixed_indices) > 0:
        # #     ts_candidate.set_constraint(FixAtoms(indices=fixed_indices.tolist()))

        # print("🔍 开始过渡态精修（不使用 Hessian）...")
        # dyn = Sella(
        #     ts_candidate,
        #     trajectory='ts.traj',
        #     logfile='ts.log',
        #     # hessian_function=get_hessian
        # )
        # dyn.run(fmax=0.05, steps=200)
        final_atoms = pre_dyn.atoms
        # except Exception as exc:
        #     print(f"⚠️ 带 Hessian 的 Sella 失败，回退到预搜索结果: {exc}")
        #     final_atoms = ts_candidate

        print(f"✅ 搜索完成，用时 {time.time() - start_time:.2f} 秒")
        
        io.write("ts_final.vasp", final_atoms)
        
        initial_atoms = self.images[0]
        product_atoms = self.images[-1]
        # initial_atoms.calc = calc
        # final_atoms.calc = calc
        
        barriers = calculate_barrier_heights(initial_atoms, final_atoms, product_atoms)
        
        with open("barriers.json", 'w') as f:
            json.dump(barriers, f, indent=2)
        
        self.checkpoint.save(Stage.SELLA_SEARCH, ts_guess_path="ts_final.vasp")

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="过渡态优化工作流")
    parser.add_argument("--is-poscar", default="POSCAR_is", help="初始结构文件")
    parser.add_argument("--fs-poscar", default="POSCAR_fs", help="终态结构文件")
    parser.add_argument("--utils-dir", default="./utils", help="VASP工具目录")
    parser.add_argument("--pretrained-ckpt", required=True, help="预训练模型路径")
    parser.add_argument("--ts-guess-idx", type=int, default=3, help="TS 猜测结构索引")
    parser.add_argument("--n-images", type=int, default=7, help="插值图像数量")
    parser.add_argument("--clear-checkpoint", action="store_true", help="清除断点")
    
    args = parser.parse_args()
    
    if args.clear_checkpoint:
        checkpoint = CheckpointManager()
        checkpoint.clear()
        print("🧹 已清除断点")
        return
    
    workflow = TSOptimizationWorkflow(args)
    workflow.run()

if __name__ == "__main__":
    main()
