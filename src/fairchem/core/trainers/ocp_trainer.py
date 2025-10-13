"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm
import pickle
from ase.data import chemical_symbols

from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.common.relaxation.ml_relaxation import ml_relax
from fairchem.core.common.utils import cg_change_mat, check_traj_files, irreps_sum
from fairchem.core.modules.evaluator import Evaluator
from fairchem.core.modules.scaling.util import ensure_fitted
from fairchem.core.trainers.base_trainer import BaseTrainer
from fairchem.core.models.gemnet_oc.qeq import QEqModule


@registry.register_trainer("ocp")
@registry.register_trainer("energy")
@registry.register_trainer("forces")
class OCPTrainer(BaseTrainer):
    """
    Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
        and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        outputs (dict): Output property configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        loss_fns (dict): Loss function configuration.
        eval_metrics (dict): Evaluation metrics configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`wandb`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
        noddp (bool, optional): Run model without DDP.
    """

    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="wandb",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm=None,
        noddp=False,
        name="ocp",
    ):
        if slurm is None:
            slurm = {}
        super().__init__(
            task=task,
            model=model,
            outputs=outputs,
            dataset=dataset,
            optimizer=optimizer,
            loss_fns=loss_fns,
            eval_metrics=eval_metrics,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
            name=name,
        )

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if not hasattr(self, "primary_metric") or self.primary_metric != primary_metric:
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            skip_steps = self.step % len(self.train_loader)
            self.train_sampler.set_epoch_and_start_iteration(epoch_int, skip_steps)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update("loss", loss.item(), self.metrics)

                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                        "w": out['w'].mean().item(),
                        "q1": out['charge'][0].item(),
                        "q2": out['charge'][1].item(),
                        "q3": out['charge'][2].item(),
                        "charge_energy": out['charge_energy'].item(),
                        "lambda_sol": out['lambda_sol'].item(),
                    }
                )

                # Add dynamic loss weights for common targets if present
                try:
                    energy_w = self._get_dynamic_loss_weight("energy", default=None)
                    forces_w = self._get_dynamic_loss_weight("forces", default=None)
                    if energy_w is not None:
                        log_dict.update({"lossw_energy": float(energy_w)})
                    if forces_w is not None:
                        log_dict.update({"lossw_forces": float(forces_w)})
                except Exception:
                    pass

                log_dict.update(self.loss_dict)

                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                ):
                    log_str = [f"{k}: {v:.2e}" for k, v in log_dict.items()]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if checkpoint_every != -1 and self.step % checkpoint_every == 0:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics["loss"]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch):
        out = self.model(batch.to(self.device))

        ### TODO: Move into BaseModel in OCP 2.0
        outputs = {}
        batch_size = batch.natoms.numel()
        num_atoms_in_batch = batch.natoms.sum()
        for target_key in self.output_targets:
            if target_key == 'charge_energy' or target_key == 'lambda_sol':
                continue
            ### Target property is a direct output of the model
            if target_key in out:
                pred = out[target_key]
            ## Target property is a derived output of the model. Construct the
            ## parent property
            else:
                _max_rank = 0
                for subtarget_key in self.output_targets[target_key]["decomposition"]:
                    _max_rank = max(
                        _max_rank,
                        self.output_targets[subtarget_key]["irrep_dim"],
                    )

                pred_irreps = torch.zeros(
                    (batch_size, irreps_sum(_max_rank)), device=self.device
                )

                for subtarget_key in self.output_targets[target_key]["decomposition"]:
                    irreps = self.output_targets[subtarget_key]["irrep_dim"]
                    _pred = out[subtarget_key]

                    if self.normalizers.get(subtarget_key, False):
                        _pred = self.normalizers[subtarget_key].denorm(_pred)

                    ## Fill in the corresponding irreps prediction
                    ## Reshape irrep prediction to (batch_size, irrep_dim)
                    pred_irreps[
                        :,
                        max(0, irreps_sum(irreps - 1)) : irreps_sum(irreps),
                    ] = _pred.view(batch_size, -1)

                pred = torch.einsum(
                    "ba, cb->ca",
                    cg_change_mat(_max_rank, self.device),
                    pred_irreps,
                )

            ### not all models are consistent with the output shape
            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_key]["level"] == "atom":
                pred = pred.view(num_atoms_in_batch, -1)
            else:
                pred = pred.view(batch_size, -1)

            outputs[target_key] = pred
        outputs['charge'] = out.get('charge', None)
        outputs['charge_energy'] = out.get('charge_energy', None)
        outputs['w'] = out.get('w', None)
        outputs['lambda_sol'] = out.get('lambda_sol', None)
        # outputs['qeq_force'] = out.get('qeq_force', None)

        return outputs

    def _compute_loss(self, out, batch):
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0
        self.loss_dict = {}
        loss = []
        for loss_fn in self.loss_fns:
            target_name, loss_info = loss_fn

            target = batch[target_name]
            pred = out[target_name]

            natoms = batch.natoms
            natoms = torch.repeat_interleave(natoms, natoms)

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["train_on_free_atoms"]
            ):
                target = target[mask]
                pred = pred[mask]
                natoms = natoms[mask]

            num_atoms_in_batch = natoms.numel()
            # if target_name == 'energy':

            #     data = pickle.load(open('avge0.pkl', 'rb'))
            #     target = target - np.sum([data[i] for i in batch.atomic_numbers.to(torch.int16)])

            if self.normalizers.get(target_name, False):
                target = self.normalizers[target_name].norm(target)

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
            else:
                target = target.view(batch_size, -1)

            # Base static multiplier from config
            base_mult = loss_info["coefficient"]
            # Optional dynamic schedule override
            mult = self._get_dynamic_loss_weight(target_name, default=base_mult)
            self.loss_dict[target_name] = mult \
                * loss_info["fn"](
                    pred,
                    target,
                    natoms=natoms,
                    batch_size=batch_size,
                )
            
            
            loss.append(
                self.loss_dict[target_name]
            )
        calc_qeq = False
        calc_charge = False
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
            # if target_name == 'charge':
        
        def get_symbol_by_number(atomic_number):
            return chemical_symbols[atomic_number]
        
        def electronegativity_rank_loss(output, atoms, en_dict):
            from collections import defaultdict

            charges = output.squeeze()  # 假设输出形状为 [N]
            atom_groups = defaultdict(list)

            # 构建原子索引映射：按元素分组
            for i, atom in enumerate(atoms):
                symbol = get_symbol_by_number(atom)
                atom_groups[symbol].append(i)

            avg_charges = []
            en_values = []

            # 对每个元素计算平均电荷，并获取电负性
            for symbol, indices in atom_groups.items():
                group_charges = charges[indices]  # 获取该元素的所有电荷值
                avg_charge = torch.mean(group_charges)  # 平均电荷（保留梯度）
                avg_charges.append(avg_charge)
                en_values.append(en_dict[symbol])

            # 转换为张量
            avg_charges_tensor = torch.stack(avg_charges)
            en_values_tensor = torch.tensor(en_values, device=charges.device)

            # 排序并获取秩（rank）
            _, charge_order = torch.sort(avg_charges_tensor, stable=True)
            _, en_order = torch.sort(en_values_tensor, stable=True)

            # 创建排序 -> 秩 的映射
            charge_ranks = torch.argsort(charge_order, stable=True).float()
            en_ranks = torch.argsort(en_order, stable=True).float()

            # 计算 Spearman 相关系数
            corr = torch.corrcoef(torch.stack([en_ranks, charge_ranks]))[0, 1]

            # 损失定义为 1 - 相关系数
            loss = 1.0 - corr

            return loss
        
        if calc_qeq:
        #     # eqemodel = self.model.qeq_module
        #     # grad_outputs = torch.ones_like(out['charge_energy'])
        #     # out['qeq_force'] = -1 * eqemodel.get_qeq_force(out['charge_energy'], out['pre_charge'], grad_outputs=grad_outputs)
            
        #     # loss.append(300 * out['qeq_force'])

        #     # self.loss_dict['en_loss'] = 1000 * electronegativity_rank_loss(out['pre_charge'], batch.atomic_numbers.to(torch.int16), en_dict=en_dict)
        #     # loss.append(self.loss_dict['en_loss'])

        #     # self.loss_dict['qeq_loss'] = out['qeq_force'] * 300
        #     # loss.append()
        #     # self.loss_dict['en_loss'] = electronegativity_rank_loss(out['charge'], batch.atomic_numbers.to(torch.int16), en_dict=en_dict)
            loss_w = torch.mean(torch.abs(out['w'] - 4.44 - batch.mu ))
            self.loss_dict['loss_w'] = loss_w * 200
        #     loss.append(loss_w)
            loss.append(loss_w * 200)
            # self.loss_dict['loss_w'] = loss_w * 1000
        # Sanity check to make sure the compute graph is correct.
        if calc_charge:
            loss_charge = torch.mean(torch.abs(out['charge'] - torch.tensor(batch.bader[0], device=out['charge'].device, dtype=out['charge'].dtype)))  # 这里的bader是个数组
            self.loss_dict['loss_charge'] = loss_charge * 100
            loss.append(loss_charge * 100)
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        return sum(loss)

    def _get_dynamic_loss_weight(self, target_name, default=None):
        """
        Compute a dynamic loss weight for a given target based on the configured
        schedule in self.config["optim"]["loss_schedule"]. If no schedule for the
        target exists, returns `default`.

        Supported schedules (example under optim.loss_schedule):
          energy: {type: "linear", start: 0.0, end: 1.0, start_step: 0, end_step: 10000}
          forces: {type: "cosine", start: 1.0, end: 0.2, duration: 50000, start_step: 0}
          energy: {type: "step", values: [[0, 0.0], [10000, 0.5], [50000, 1.0]]}
          forces: {type: "exp", start: 1.0, gamma: 0.9995, start_step: 0}
        """
        sched_cfg = {}
        final_config = {}
        for config in self.config.get("loss_fns", {}):
            final_config.update(config)
        sched_cfg = final_config.get(target_name, {})
        if not sched_cfg:
            return default

        step = int(getattr(self, "step", 0))
        sched_type = str(sched_cfg.get("type", "linear")).lower()

        def clip01(x):
            return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

        if sched_type == "linear":
            start = float(sched_cfg.get("start", default if default is not None else 1.0))
            end = float(sched_cfg.get("end", start))
            s0 = int(sched_cfg.get("start_step", 0))
            s1 = int(sched_cfg.get("end_step", s0))
            if s1 <= s0:
                return end
            t = clip01((step - s0) / float(max(1, s1 - s0)))
            return (1 - t) * start + t * end

        if sched_type == "cosine":
            import math
            start = float(sched_cfg.get("start", default if default is not None else 1.0))
            end = float(sched_cfg.get("end", start))
            s0 = int(sched_cfg.get("start_step", 0))
            dur = int(sched_cfg.get("duration", 1))
            t = clip01((step - s0) / float(max(1, dur)))
            return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))

        if sched_type == "step":
            # values: list of [at_step, value], sorted by at_step
            values = sched_cfg.get("values", [])
            if not values:
                return default
            current = float(values[0][1])
            for at, val in values:
                if step >= int(at):
                    current = float(val)
                else:
                    break
            return current

        if sched_type in ("exp", "exponential"):
            start = float(sched_cfg.get("start", default if default is not None else 1.0))
            gamma = float(sched_cfg.get("gamma", 1.0))
            s0 = int(sched_cfg.get("start_step", 0))
            k = max(0, step - s0)
            return start * (gamma ** k)

        # Fallback
        return default

    def _compute_metrics(self, out, batch, evaluator, metrics=None):
        if metrics is None:
            metrics = {}
        # this function changes the values in the out dictionary,
        # make a copy instead of changing them in the callers version
        out = {k: v.clone() for k, v in out.items()}

        natoms = batch.natoms
        batch_size = natoms.numel()

        ### Retrieve free atoms
        fixed = batch.fixed
        mask = fixed == 0

        s_idx = 0
        natoms_free = []
        for _natoms in natoms:
            natoms_free.append(torch.sum(mask[s_idx : s_idx + _natoms]).item())
            s_idx += _natoms
        natoms = torch.LongTensor(natoms_free).to(self.device)

        targets = {}
        for target_name in self.output_targets:
            if target_name == "charge":
                continue

            if target_name == "w":
                continue
                
            target = batch[target_name]
            num_atoms_in_batch = batch.natoms.sum()

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["eval_on_free_atoms"]
            ):
                target = target[mask]
                out[target_name] = out[target_name][mask]
                num_atoms_in_batch = natoms.sum()

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
            else:
                target = target.view(batch_size, -1)

            targets[target_name] = target
            if self.normalizers.get(target_name, False):
                out[target_name] = self.normalizers[target_name].denorm(
                    out[target_name]
                )

        targets["natoms"] = natoms
        out["natoms"] = natoms

        

        return evaluator.eval(out, targets, prev_metrics=metrics)

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file: str | None = None,
        disable_tqdm: bool = False,
    ):
        if self.is_debug and per_image:
            raise FileNotFoundError("Predictions require debug mode to be turned off.")

        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [data_loader]

        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        predictions = defaultdict(list)

        for _i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            for target_key in self.config['outputs']:
                pred = out[target_key]
                if self.normalizers.get(target_key, False):
                    pred = self.normalizers[target_key].denorm(pred)
                
                # if target_key == "energy":
                #     data = pickle.load(open('avge0.pkl', 'rb'))
                #     pred = pred + np.sum([data[i] for i in batch.atomic_numbers.to(torch.int16)])

                if per_image:
                    ### Save outputs in desired precision, default float16
                    if (
                        self.config["outputs"][target_key].get(
                            "prediction_dtype", "float16"
                        )
                        == "float32"
                        or self.config["task"].get("prediction_dtype", "float16")
                        == "float32"
                        or self.config["task"].get("dataset", "lmdb") == "oc22_lmdb"
                    ):
                        dtype = torch.float32
                    else:
                        dtype = torch.float16

                    pred = pred.cpu().detach().to(dtype)
                    ### Split predictions into per-image predictions
                    if self.config["outputs"][target_key]["level"] == "atom":
                        batch_natoms = batch.natoms
                        batch_fixed = batch.fixed
                        per_image_pred = torch.split(pred, batch_natoms.tolist())

                        ### Save out only free atom, EvalAI does not need fixed atoms
                        _per_image_fixed = torch.split(
                            batch_fixed, batch_natoms.tolist()
                        )
                        _per_image_free_preds = [
                            _pred[(fixed == 0).tolist()].numpy()
                            for _pred, fixed in zip(per_image_pred, _per_image_fixed)
                        ]
                        _chunk_idx = np.array(
                            [free_pred.shape[0] for free_pred in _per_image_free_preds]
                        )
                        per_image_pred = _per_image_free_preds
                    ### Assumes system level properties are of the same dimension
                    else:
                        per_image_pred = pred.numpy()
                        _chunk_idx = None

                    predictions[f"{target_key}"].extend(per_image_pred)
                    ### Backwards compatibility, retain 'chunk_idx' for forces.
                    if _chunk_idx is not None:
                        if target_key == "forces":
                            predictions["chunk_idx"].extend(_chunk_idx)
                        else:
                            predictions[f"{target_key}_chunk_idx"].extend(_chunk_idx)
                else:
                    predictions[f"{target_key}"] = pred.detach()

            if not per_image:
                return predictions

            ### Get unique system identifiers
            sids = (
                batch.sid.tolist() if isinstance(batch.sid, torch.Tensor) else batch.sid
            )
            ## Support naming structure for OC20 S2EF
            if "fid" in batch:
                fids = (
                    batch.fid.tolist()
                    if isinstance(batch.fid, torch.Tensor)
                    else batch.fid
                )
                systemids = [f"{sid}_{fid}" for sid, fid in zip(sids, fids)]
            else:
                systemids = [f"{sid}" for sid in sids]

            predictions["ids"].extend(systemids)

        self.save_results(predictions, results_file)

        if self.ema:
            self.ema.restore()

        return predictions

    def run_relaxations(self, split="val"):
        ensure_fitted(self._unwrapped_model)

        # When set to true, uses deterministic CUDA scatter ops, if available.
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        # Only implemented for GemNet-OC currently.
        registry.register(
            "set_deterministic_scatter",
            self.config["task"].get("set_deterministic_scatter", False),
        )

        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
        # Else just generate predictions.
        if (
            hasattr(self.relax_dataset[0], "pos_relaxed")
            and self.relax_dataset[0].pos_relaxed is not None
        ) and (
            hasattr(self.relax_dataset[0], "y_relaxed")
            and self.relax_dataset[0].y_relaxed is not None
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                logging.info(
                    f"Skipping batch: {batch.sid.tolist() if isinstance(batch.sid, torch.Tensor) else batch.sid}"
                )
                continue

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 300),
                fmax=self.config["task"].get("relaxation_fmax", 0.02),
                relax_opt=self.config["task"]["relax_opt"],
                save_full_traj=self.config["task"].get("save_full_traj", True),
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                sid_list = (
                    relaxed_batch.sid.tolist()
                    if isinstance(relaxed_batch.sid, torch.Tensor)
                    else relaxed_batch.sid
                )
                systemids = [str(sid) for sid in sid_list]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(torch.sum(mask[s_idx : s_idx + natoms]).item())
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

        if self.config["task"].get("write_pos", False):
            results = distutils.gather_objects(
                {"ids": ids, "pos": relaxed_positions, "chunk_idx": chunk_idx}
            )
            distutils.synchronize()
            if distutils.is_master():
                gather_results = {
                    key: list(chain(*(result[key] for result in results)))
                    for key in results[0]
                }

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    [gather_results["pos"][i] for i in idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    [gather_results["chunk_idx"][i] for i in idx]
                )[:-1]  # np.split does not need last idx, assumes n-1:end

                full_path = os.path.join(
                    self.config["cmd"]["results_dir"], "relaxed_positions.npz"
                )
                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {f"{task}_{k}": metrics[k]["metric"] for k in metrics}
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()

        registry.unregister("set_deterministic_scatter")
