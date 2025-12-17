#!/usr/bin/env python3
"""
autodE集成模块 - 自动过渡态（TS）计算

使用autodE自动计算纳米酶催化反应的过渡态，提供精确的活化能和反应路径信息。
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import autode as ade
    AUTODE_AVAILABLE = True
except ImportError:
    AUTODE_AVAILABLE = False
    # 创建一个虚拟的 ade 模块以避免 NameError
    class _DummyAde:
        class Molecule: pass
        class Reaction: pass
        class Config:
            n_cores = 1
            max_core = 3600
            lcode = 'xtb'
            hcode = 'xtb'
        class methods:
            class XTB: pass
    ade = _DummyAde()
    print("Warning: autodE not installed. TS calculation will be disabled.")
    print("Install with: pip install autode")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None


class AutodETSCalculator:
    """
    autodE过渡态计算器

    功能：
    1. 自动搜索过渡态结构
    2. 计算活化能（Ea）
    3. 计算反应能（ΔE）
    4. 生成反应路径（IRC）
    5. 验证过渡态（频率分析）
    """

    def __init__(
        self,
        method: str = 'xtb',  # 'xtb', 'orca', 'g09', 'g16'
        n_cores: int = 4,
        max_time: int = 3600,  # 最大计算时间（秒）
        temp_dir: Optional[str] = None
    ):
        """
        初始化autodE计算器

        Args:
            method: 计算方法 ('xtb'最快, 'orca'精确)
            n_cores: CPU核心数
            max_time: 单个计算最大时间（秒）
            temp_dir: 临时文件目录
        """
        if not AUTODE_AVAILABLE:
            raise ImportError("autodE not installed. Install with: pip install autode")

        self.method = method
        self.n_cores = n_cores
        self.max_time = max_time
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix='autode_')

        # 配置autodE
        ade.Config.n_cores = n_cores
        ade.Config.max_core = max_time

        # 设置计算方法
        if method == 'xtb':
            ade.Config.lcode = 'xtb'  # 低级别方法
            ade.Config.hcode = 'xtb'  # 高级别方法
        elif method == 'orca':
            ade.Config.lcode = 'xtb'
            ade.Config.hcode = 'orca'

        print(f"✓ autodE initialized with method={method}, n_cores={n_cores}")

    def calculate_reaction_profile(
        self,
        nanozyme_xyz: str,
        substrate_smiles: str,
        product_smiles: Optional[str] = None,
        charge: int = 0,
        mult: int = 1
    ) -> Dict[str, Any]:
        """
        计算完整的反应能量曲线

        Args:
            nanozyme_xyz: 纳米酶XYZ坐标文件路径
            substrate_smiles: 底物SMILES
            product_smiles: 产物SMILES（可选，自动推断）
            charge: 总电荷
            mult: 自旋多重度

        Returns:
            {
                'activation_energy': float,  # 活化能 (kcal/mol)
                'reaction_energy': float,    # 反应能 (kcal/mol)
                'ts_structure': str,         # TS结构XYZ
                'ts_frequency': float,       # 虚频 (cm^-1)
                'irc_path': List[Dict],      # IRC路径
                'success': bool
            }
        """
        print(f"\n{'='*60}")
        print(f"计算反应能量曲线: {substrate_smiles}")
        print(f"{'='*60}")

        try:
            # 1. 构建反应物复合物
            reactant = self._build_reactant_complex(nanozyme_xyz, substrate_smiles, charge, mult)
            print(f"✓ 反应物复合物构建完成")

            # 2. 构建产物（如果未提供则自动推断）
            if product_smiles is None:
                product_smiles = self._infer_product(substrate_smiles)

            product = self._build_product_complex(nanozyme_xyz, product_smiles, charge, mult)
            print(f"✓ 产物复合物构建完成")

            # 3. 定义反应
            reaction = ade.Reaction(reactant, product, name='nanozyme_catalysis')
            print(f"✓ 反应定义完成")

            # 4. 计算反应路径（自动搜索TS）
            print(f"⏳ 搜索过渡态中...")
            reaction.calculate_reaction_profile()

            # 5. 提取结果
            if reaction.ts is None:
                print("✗ 未找到过渡态")
                return {'success': False, 'error': 'TS not found'}

            # 活化能 (kcal/mol)
            ea_forward = reaction.ts.energy - reactant.energy
            ea_forward_kcal = ea_forward * 627.509  # Hartree to kcal/mol

            # 反应能
            delta_e = product.energy - reactant.energy
            delta_e_kcal = delta_e * 627.509

            # TS频率分析
            ts_freq = self._get_imaginary_frequency(reaction.ts)

            # IRC路径
            irc_path = self._calculate_irc(reaction.ts) if hasattr(reaction, 'ts') else []

            print(f"\n{'='*60}")
            print(f"✓ 计算完成!")
            print(f"  活化能 (Ea):     {ea_forward_kcal:.2f} kcal/mol")
            print(f"  反应能 (ΔE):     {delta_e_kcal:.2f} kcal/mol")
            print(f"  TS虚频:          {ts_freq:.1f} cm⁻¹")
            print(f"{'='*60}\n")

            return {
                'activation_energy': ea_forward_kcal,
                'reaction_energy': delta_e_kcal,
                'ts_structure': self._get_xyz_string(reaction.ts),
                'ts_frequency': ts_freq,
                'ts_energy': reaction.ts.energy * 627.509,
                'reactant_energy': reactant.energy * 627.509,
                'product_energy': product.energy * 627.509,
                'irc_path': irc_path,
                'success': True
            }

        except Exception as e:
            print(f"✗ 计算失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def quick_estimate_barrier(
        self,
        nanozyme_xyz: str,
        substrate_smiles: str,
        use_ml: bool = True
    ) -> float:
        """
        快速估算活化能（用于大规模筛选）

        使用经验规则 + ML模型快速估算，避免昂贵的TS搜索

        Args:
            nanozyme_xyz: 纳米酶结构
            substrate_smiles: 底物SMILES
            use_ml: 是否使用ML模型

        Returns:
            estimated_ea: 估算的活化能 (kcal/mol)
        """
        # 方法1: 基于几何的快速估算
        geometric_ea = self._geometric_barrier_estimate(nanozyme_xyz, substrate_smiles)

        if not use_ml:
            return geometric_ea

        # 方法2: ML模型预测（如果可用）
        try:
            ml_ea = self._ml_barrier_prediction(nanozyme_xyz, substrate_smiles)
            # 加权平均
            return 0.6 * ml_ea + 0.4 * geometric_ea
        except:
            return geometric_ea

    def _build_reactant_complex(
        self,
        nanozyme_xyz: str,
        substrate_smiles: str,
        charge: int,
        mult: int
    ) -> Any:  # 使用 Any 替代 ade.Molecule 以避免导入错误
        """构建反应物复合物（纳米酶 + 底物）"""
        # 读取纳米酶结构
        nanozyme = ade.Molecule(nanozyme_xyz, charge=charge, mult=mult)

        # 从SMILES构建底物
        substrate = ade.Molecule(smiles=substrate_smiles, charge=0, mult=1)
        substrate.optimise(method=ade.methods.XTB())

        # 将底物放置在纳米酶活性位点附近
        complex_mol = self._dock_substrate(nanozyme, substrate)

        # 优化复合物
        complex_mol.optimise(method=ade.methods.XTB())

        return complex_mol

    def _build_product_complex(
        self,
        nanozyme_xyz: str,
        product_smiles: str,
        charge: int,
        mult: int
    ) -> ade.Molecule:
        """构建产物复合物"""
        nanozyme = ade.Molecule(nanozyme_xyz, charge=charge, mult=mult)
        product = ade.Molecule(smiles=product_smiles, charge=0, mult=1)
        product.optimise(method=ade.methods.XTB())

        complex_mol = self._dock_substrate(nanozyme, product)
        complex_mol.optimise(method=ade.methods.XTB())

        return complex_mol

    def _dock_substrate(
        self,
        nanozyme: ade.Molecule,
        substrate: ade.Molecule,
        distance: float = 3.0
    ) -> ade.Molecule:
        """
        将底物对接到纳米酶活性位点

        简单策略：将底物放置在纳米酶几何中心上方
        """
        # 计算纳米酶中心
        nanozyme_coords = np.array([atom.coord for atom in nanozyme.atoms])
        center = nanozyme_coords.mean(axis=0)

        # 将底物移动到中心上方
        substrate_coords = np.array([atom.coord for atom in substrate.atoms])
        substrate_center = substrate_coords.mean(axis=0)

        offset = center + np.array([0, 0, distance]) - substrate_center

        for atom in substrate.atoms:
            atom.coord = atom.coord + offset

        # 合并分子
        complex_mol = nanozyme.copy()
        for atom in substrate.atoms:
            complex_mol.atoms.append(atom)

        return complex_mol

    def _infer_product(self, substrate_smiles: str) -> str:
        """
        根据底物自动推断产物

        针对常见反应类型的简单规则
        """
        # TMB氧化: TMB -> TMB(ox)
        if 'TMB' in substrate_smiles or 'c1ccc(N)cc1' in substrate_smiles:
            return 'c1ccc([N+](=O)[O-])cc1'  # 硝基苯

        # pNPP水解: pNPP -> pNP + Pi
        if 'OP(=O)(O)Oc1ccc([N+](=O)[O-])cc1' in substrate_smiles:
            return 'Oc1ccc([N+](=O)[O-])cc1'  # 对硝基苯酚

        # H2O2分解: H2O2 -> H2O + 1/2 O2
        if substrate_smiles == 'OO':
            return 'O'  # 水

        # 默认：返回原始SMILES（无反应）
        return substrate_smiles

    def _get_imaginary_frequency(self, ts: ade.Molecule) -> float:
        """获取TS的虚频"""
        if hasattr(ts, 'imaginary_frequencies') and ts.imaginary_frequencies:
            return ts.imaginary_frequencies[0]
        return 0.0

    def _get_xyz_string(self, molecule: ade.Molecule) -> str:
        """将分子转换为XYZ字符串"""
        lines = [f"{len(molecule.atoms)}"]
        lines.append(f"TS structure")
        for atom in molecule.atoms:
            x, y, z = atom.coord
            lines.append(f"{atom.label:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
        return '\n'.join(lines)

    def _calculate_irc(self, ts: ade.Molecule) -> List[Dict]:
        """
        计算内禀反应坐标（IRC）

        Returns:
            List of {'coords': np.array, 'energy': float}
        """
        # 简化版：只返回TS点
        # 完整IRC需要更复杂的实现
        return [{
            'step': 0,
            'energy': ts.energy * 627.509,
            'coords': np.array([atom.coord for atom in ts.atoms])
        }]

    def _geometric_barrier_estimate(
        self,
        nanozyme_xyz: str,
        substrate_smiles: str
    ) -> float:
        """
        基于几何的快速活化能估算

        使用经验公式：Ea ≈ f(距离, 角度, 电荷)
        """
        # 读取纳米酶结构
        with open(nanozyme_xyz, 'r') as f:
            lines = f.readlines()

        # 提取催化中心原子
        # 简化：假设前3个原子是催化中心
        coords = []
        for line in lines[2:5]:  # 跳过前两行
            parts = line.split()
            if len(parts) >= 4:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if len(coords) < 2:
            return 20.0  # 默认值

        # 计算催化中心间距
        coords = np.array(coords)
        distances = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)

        avg_dist = np.mean(distances)

        # 经验公式：距离越近，活化能越低
        # Ea ≈ 25 - 5 * (1 / avg_dist)
        ea_estimate = max(5.0, 25.0 - 5.0 / avg_dist)

        return ea_estimate

    def calculate_single_point_energy_xtb(
        self,
        xyz_file: str,
        charge: int = 0,
        mult: int = 1
    ) -> Dict[str, Any]:
        """
        使用xTB直接计算单点能量

        Args:
            xyz_file: XYZ结构文件
            charge: 总电荷
            mult: 自旋多重度

        Returns:
            {
                'energy': float (Hartree),
                'energy_kcal': float (kcal/mol),
                'success': bool
            }
        """
        try:
            mol = ade.Molecule(xyz_file, charge=charge, mult=mult)

            # 使用xTB计算单点能量
            mol.single_point(method=ade.methods.XTB())

            energy_hartree = mol.energy
            energy_kcal = energy_hartree * 627.509  # Hartree to kcal/mol

            return {
                'energy': energy_hartree,
                'energy_kcal': energy_kcal,
                'success': True
            }
        except Exception as e:
            print(f"xTB单点能量计算失败: {e}")
            return {'success': False, 'error': str(e)}

    def optimize_structure_xtb(
        self,
        xyz_file: str,
        output_file: str,
        charge: int = 0,
        mult: int = 1
    ) -> Dict[str, Any]:
        """
        使用xTB优化分子结构

        Args:
            xyz_file: 输入XYZ文件
            output_file: 输出优化后的XYZ文件
            charge: 总电荷
            mult: 自旋多重度

        Returns:
            {
                'optimized_energy': float (kcal/mol),
                'optimized_structure': str (XYZ格式),
                'success': bool
            }
        """
        try:
            mol = ade.Molecule(xyz_file, charge=charge, mult=mult)

            # 使用xTB优化结构
            print(f"⏳ 使用xTB优化结构...")
            mol.optimise(method=ade.methods.XTB())

            # 保存优化后的结构
            with open(output_file, 'w') as f:
                f.write(self._get_xyz_string(mol))

            energy_kcal = mol.energy * 627.509

            print(f"✓ 优化完成! 能量: {energy_kcal:.2f} kcal/mol")

            return {
                'optimized_energy': energy_kcal,
                'optimized_structure': self._get_xyz_string(mol),
                'output_file': output_file,
                'success': True
            }
        except Exception as e:
            print(f"xTB结构优化失败: {e}")
            return {'success': False, 'error': str(e)}

    def calculate_reaction_barrier_xtb(
        self,
        reactant_xyz: str,
        product_xyz: str,
        charge: int = 0,
        mult: int = 1
    ) -> Dict[str, Any]:
        """
        使用xTB计算反应能垒（简化版）

        通过优化反应物和产物，估算能垒

        Args:
            reactant_xyz: 反应物XYZ文件
            product_xyz: 产物XYZ文件
            charge: 总电荷
            mult: 自旋多重度

        Returns:
            {
                'reactant_energy': float (kcal/mol),
                'product_energy': float (kcal/mol),
                'reaction_energy': float (kcal/mol),
                'estimated_barrier': float (kcal/mol),
                'success': bool
            }
        """
        try:
            # 优化反应物
            print("⏳ 优化反应物...")
            reactant = ade.Molecule(reactant_xyz, charge=charge, mult=mult)
            reactant.optimise(method=ade.methods.XTB())
            e_reactant = reactant.energy * 627.509

            # 优化产物
            print("⏳ 优化产物...")
            product = ade.Molecule(product_xyz, charge=charge, mult=mult)
            product.optimise(method=ade.methods.XTB())
            e_product = product.energy * 627.509

            # 反应能
            delta_e = e_product - e_reactant

            # 估算能垒（使用Hammond假设）
            # 对于放热反应：Ea ≈ 10-15 kcal/mol
            # 对于吸热反应：Ea ≈ ΔE + 10-15 kcal/mol
            if delta_e < 0:  # 放热
                estimated_barrier = 12.0
            else:  # 吸热
                estimated_barrier = delta_e + 12.0

            print(f"\n{'='*60}")
            print(f"✓ xTB反应能垒计算完成")
            print(f"  反应物能量: {e_reactant:.2f} kcal/mol")
            print(f"  产物能量:   {e_product:.2f} kcal/mol")
            print(f"  反应能 (ΔE): {delta_e:.2f} kcal/mol")
            print(f"  估算能垒:   {estimated_barrier:.2f} kcal/mol")
            print(f"{'='*60}\n")

            return {
                'reactant_energy': e_reactant,
                'product_energy': e_product,
                'reaction_energy': delta_e,
                'estimated_barrier': estimated_barrier,
                'success': True
            }
        except Exception as e:
            print(f"xTB能垒计算失败: {e}")
            return {'success': False, 'error': str(e)}

    def _ml_barrier_prediction(
        self,
        nanozyme_xyz: str,
        substrate_smiles: str
    ) -> float:
        """
        使用ML模型预测活化能

        TODO: 训练一个GNN模型预测Ea
        """
        # 占位符：返回几何估算
        return self._geometric_barrier_estimate(nanozyme_xyz, substrate_smiles)


class SubstrateReactionLibrary:
    """
    底物反应库 - 预定义常见底物的反应参数
    """

    REACTIONS = {
        'TMB': {
            'name': 'TMB氧化',
            'substrate_smiles': 'c1ccc(N)cc1',
            'product_smiles': 'c1ccc([N+](=O)[O-])cc1',
            'reaction_type': 'oxidation',
            'typical_ea': 15.0,  # kcal/mol
            'charge': 0,
            'mult': 1
        },
        'pNPP': {
            'name': 'pNPP水解',
            'substrate_smiles': 'OP(=O)(O)Oc1ccc([N+](=O)[O-])cc1',
            'product_smiles': 'Oc1ccc([N+](=O)[O-])cc1',
            'reaction_type': 'hydrolysis',
            'typical_ea': 18.0,
            'charge': -1,
            'mult': 1
        },
        'ABTS': {
            'name': 'ABTS氧化',
            'substrate_smiles': 'c1ccc(S(=O)(=O)Nc2ccc(N=Nc3ccc(NS(=O)(=O)c4ccccc4)cc3)cc2)cc1',
            'product_smiles': 'c1ccc(S(=O)(=O)[N+](=O)c2ccc(N=Nc3ccc([N+](=O)S(=O)(=O)c4ccccc4)cc3)cc2)cc1',
            'reaction_type': 'oxidation',
            'typical_ea': 16.0,
            'charge': -2,
            'mult': 1
        },
        'H2O2': {
            'name': 'H2O2分解',
            'substrate_smiles': 'OO',
            'product_smiles': 'O',
            'reaction_type': 'decomposition',
            'typical_ea': 12.0,
            'charge': 0,
            'mult': 1
        },
        'OPD': {
            'name': 'OPD氧化',
            'substrate_smiles': 'c1ccc(N)c(N)c1',
            'product_smiles': 'c1ccc([N+](=O)[O-])c([N+](=O)[O-])c1',
            'reaction_type': 'oxidation',
            'typical_ea': 14.0,
            'charge': 0,
            'mult': 1
        },
        'GSH': {
            'name': 'GSH氧化',
            'substrate_smiles': 'NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O',
            'product_smiles': 'NC(CCC(=O)NC(CSSC(C(=O)NCC(=O)O)NC(=O)CCC(N)C(=O)O)C(=O)NCC(=O)O)C(=O)O',
            'reaction_type': 'oxidation',
            'typical_ea': 17.0,
            'charge': -1,
            'mult': 1
        }
    }

    @classmethod
    def get_reaction_params(cls, substrate_name: str) -> Dict:
        """获取底物反应参数"""
        return cls.REACTIONS.get(substrate_name.upper(), None)

    @classmethod
    def list_substrates(cls) -> List[str]:
        """列出所有支持的底物"""
        return list(cls.REACTIONS.keys())


def batch_calculate_barriers(
    nanozyme_list: List[str],
    substrate: str = 'TMB',
    method: str = 'xtb',
    n_cores: int = 4,
    quick_mode: bool = False
) -> List[Dict]:
    """
    批量计算多个纳米酶的活化能

    Args:
        nanozyme_list: 纳米酶XYZ文件路径列表
        substrate: 底物名称
        method: 计算方法
        n_cores: CPU核心数
        quick_mode: 快速模式（仅估算）

    Returns:
        List of results
    """
    calculator = AutodETSCalculator(method=method, n_cores=n_cores)
    reaction_params = SubstrateReactionLibrary.get_reaction_params(substrate)

    if reaction_params is None:
        raise ValueError(f"Unknown substrate: {substrate}")

    results = []

    for i, nanozyme_xyz in enumerate(nanozyme_list):
        print(f"\n[{i+1}/{len(nanozyme_list)}] Processing: {nanozyme_xyz}")

        if quick_mode:
            # 快速估算
            ea = calculator.quick_estimate_barrier(
                nanozyme_xyz,
                reaction_params['substrate_smiles']
            )
            result = {
                'nanozyme': nanozyme_xyz,
                'substrate': substrate,
                'activation_energy': ea,
                'method': 'quick_estimate',
                'success': True
            }
        else:
            # 完整TS计算
            result = calculator.calculate_reaction_profile(
                nanozyme_xyz,
                reaction_params['substrate_smiles'],
                reaction_params['product_smiles'],
                reaction_params['charge'],
                reaction_params['mult']
            )
            result['nanozyme'] = nanozyme_xyz
            result['substrate'] = substrate

        results.append(result)

    return results


if __name__ == "__main__":
    # 测试代码
    print("autodE TS Calculator - Test")
    print("="*60)

    if not AUTODE_AVAILABLE:
        print("autodE not installed. Please install:")
        print("  pip install autode")
        print("  conda install -c conda-forge xtb")
    else:
        print("✓ autodE available")
        print(f"✓ Supported substrates: {SubstrateReactionLibrary.list_substrates()}")
