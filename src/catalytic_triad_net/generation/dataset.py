#!/usr/bin/env python3
"""
纳米酶数据集模块

支持多种分子文件格式的加载和预处理。
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
import numpy as np
import json
import logging
from pathlib import Path

from .constraints import ATOM_TO_IDX

# 导入配置系统
try:
    from ..config import get_config
except ImportError:
    get_config = None

logger = logging.getLogger(__name__)


class NanozymeDataset(Dataset):
    """
    纳米酶/催化分子数据集

    支持的文件格式：
    - XYZ: 标准XYZ格式
    - JSON: 自定义JSON格式（包含atom_types和coords字段）
    - PDB: 蛋白质数据库格式（需要Biopython）
    - MOL/SDF: MDL Molfile格式（需要RDKit）

    Args:
        data_dir: 数据目录路径
        max_atoms: 最大原子数限制
        supported_formats: 支持的文件格式列表
        center_coords: 是否中心化坐标
        validate_atoms: 是否验证原子类型
    """

    # 支持的文件格式
    SUPPORTED_FORMATS = {
        'xyz': '*.xyz',
        'json': '*.json',
        'pdb': '*.pdb',
        'mol': '*.mol',
        'sdf': '*.sdf'
    }

    def __init__(self,
                 data_dir: Union[str, Path],
                 max_atoms: Optional[int] = None,
                 supported_formats: Optional[List[str]] = None,
                 center_coords: bool = True,
                 validate_atoms: bool = True):
        self.data_dir = Path(data_dir)

        # 从配置读取参数
        if get_config:
            config = get_config()
            gen_config = config.get('generation', {})
            self.max_atoms = max_atoms or gen_config.get('max_atoms', 200)
        else:
            self.max_atoms = max_atoms or 200

        self.center_coords = center_coords
        self.validate_atoms = validate_atoms

        # 确定支持的格式
        if supported_formats is None:
            supported_formats = ['xyz', 'json']  # 默认只支持这两种
        self.supported_formats = supported_formats

        # 验证数据目录
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # 加载数据
        self.samples = self._load_data()
        logger.info(f"Loaded {len(self.samples)} samples from {self.data_dir}")

    def _load_data(self) -> List[Dict]:
        """
        加载数据文件

        Returns:
            样本列表
        """
        samples = []

        # 遍历支持的格式
        for fmt in self.supported_formats:
            if fmt not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported format: {fmt}")
                continue

            pattern = self.SUPPORTED_FORMATS[fmt]
            file_count = 0

            for filepath in self.data_dir.glob(pattern):
                file_count += 1
                try:
                    sample = self._parse_file(filepath)

                    if sample is None:
                        logger.debug(f"Skipped file (parse returned None): {filepath.name}")
                        continue

                    # 验证样本
                    if not self._validate_sample(sample, filepath):
                        continue

                    # 检查原子数限制
                    n_atoms = len(sample['atom_types'])
                    if n_atoms > self.max_atoms:
                        logger.debug(f"Skipped {filepath.name}: too many atoms ({n_atoms} > {self.max_atoms})")
                        continue

                    samples.append(sample)

                except Exception as e:
                    logger.warning(f"Failed to parse {filepath.name}: {type(e).__name__}: {e}")
                    continue

            if file_count > 0:
                logger.info(f"Processed {file_count} {fmt} files, loaded {len([s for s in samples if s.get('format') == fmt])} samples")

        if len(samples) == 0:
            logger.warning(f"No valid samples found in {self.data_dir}")

        return samples

    def _validate_sample(self, sample: Dict, filepath: Path) -> bool:
        """
        验证样本数据

        Args:
            sample: 样本字典
            filepath: 文件路径（用于错误报告）

        Returns:
            是否有效
        """
        # 检查必需字段
        if 'atom_types' not in sample or 'coords' not in sample:
            logger.warning(f"Missing required fields in {filepath.name}")
            return False

        # 检查数据一致性
        n_atoms = len(sample['atom_types'])
        n_coords = len(sample['coords'])

        if n_atoms != n_coords:
            logger.warning(f"Atom count mismatch in {filepath.name}: "
                         f"{n_atoms} atom types vs {n_coords} coordinates")
            return False

        if n_atoms == 0:
            logger.warning(f"Empty molecule in {filepath.name}")
            return False

        # 验证原子类型
        if self.validate_atoms:
            for i, atom in enumerate(sample['atom_types']):
                if atom not in ATOM_TO_IDX:
                    logger.warning(f"Unknown atom type '{atom}' at position {i} in {filepath.name}")
                    # 不拒绝，只是警告

        return True

    def _parse_file(self, filepath: Path) -> Optional[Dict]:
        """
        解析分子文件

        Args:
            filepath: 文件路径

        Returns:
            样本字典或None
        """
        suffix = filepath.suffix.lower()

        try:
            if suffix == '.xyz':
                return self._parse_xyz(filepath)
            elif suffix == '.json':
                return self._parse_json(filepath)
            elif suffix == '.pdb':
                return self._parse_pdb(filepath)
            elif suffix in ['.mol', '.sdf']:
                return self._parse_mol(filepath)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Error parsing {filepath.name}: {type(e).__name__}: {e}")
            raise

    def _parse_xyz(self, filepath: Path) -> Dict:
        """
        解析XYZ文件

        XYZ格式:
        第1行: 原子数
        第2行: 注释
        后续行: 元素符号 x y z

        Args:
            filepath: XYZ文件路径

        Returns:
            样本字典
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError(f"Invalid XYZ file: too few lines")

        try:
            n_atoms = int(lines[0].strip())
        except ValueError as e:
            raise ValueError(f"Invalid atom count in line 1: {lines[0].strip()}") from e

        if len(lines) < n_atoms + 2:
            raise ValueError(f"File has {len(lines)} lines but expects {n_atoms + 2}")

        atoms = []
        coords = []

        for line_num, line in enumerate(lines[2:2 + n_atoms], start=3):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid format at line {line_num}: expected 4 fields, got {len(parts)}")

            try:
                atom_symbol = parts[0].strip()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid data at line {line_num}: {line.strip()}") from e

            atoms.append(atom_symbol)
            coords.append([x, y, z])

        return {
            'atom_types': atoms,
            'coords': coords,
            'name': filepath.stem,
            'format': 'xyz'
        }

    def _parse_json(self, filepath: Path) -> Dict:
        """
        解析JSON文件

        期望格式:
        {
            "atom_types": ["C", "N", "O", ...],
            "coords": [[x1, y1, z1], [x2, y2, z2], ...],
            "name": "molecule_name" (可选)
        }

        Args:
            filepath: JSON文件路径

        Returns:
            样本字典
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # 确保有必需字段
        if 'atom_types' not in data or 'coords' not in data:
            raise ValueError("JSON must contain 'atom_types' and 'coords' fields")

        # 添加元数据
        if 'name' not in data:
            data['name'] = filepath.stem
        if 'format' not in data:
            data['format'] = 'json'

        return data

    def _parse_pdb(self, filepath: Path) -> Optional[Dict]:
        """
        解析PDB文件（需要Biopython）

        Args:
            filepath: PDB文件路径

        Returns:
            样本字典或None
        """
        try:
            from Bio.PDB import PDBParser
        except ImportError:
            logger.error("Biopython not installed. Cannot parse PDB files.")
            return None

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(filepath.stem, filepath)

        atoms = []
        coords = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atoms.append(atom.element)
                        coords.append(atom.coord.tolist())

        return {
            'atom_types': atoms,
            'coords': coords,
            'name': filepath.stem,
            'format': 'pdb'
        }

    def _parse_mol(self, filepath: Path) -> Optional[Dict]:
        """
        解析MOL/SDF文件（需要RDKit）

        Args:
            filepath: MOL/SDF文件路径

        Returns:
            样本字典或None
        """
        try:
            from rdkit import Chem
        except ImportError:
            logger.error("RDKit not installed. Cannot parse MOL/SDF files.")
            return None

        mol = Chem.MolFromMolFile(str(filepath))
        if mol is None:
            raise ValueError("Failed to parse MOL file")

        atoms = []
        coords = []

        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            atoms.append(atom.GetSymbol())
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

        return {
            'atom_types': atoms,
            'coords': coords,
            'name': filepath.stem,
            'format': 'mol'
        }

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含atom_types和coords的字典
        """
        sample = self.samples[idx]

        # 转换原子类型到索引
        atom_indices = []
        for atom in sample['atom_types']:
            if atom in ATOM_TO_IDX:
                atom_indices.append(ATOM_TO_IDX[atom])
            else:
                # 未知原子类型，使用碳作为默认值
                logger.debug(f"Unknown atom type '{atom}', using C as default")
                atom_indices.append(ATOM_TO_IDX.get('C', 1))

        # 转换坐标
        coords = np.array(sample['coords'], dtype=np.float32)

        # 断言：确保坐标形状正确
        assert coords.shape[0] == len(atom_indices), \
            f"Coordinate count mismatch: {coords.shape[0]} vs {len(atom_indices)}"
        assert coords.shape[1] == 3, \
            f"Coordinates must be 3D, got shape {coords.shape}"

        # 中心化坐标
        if self.center_coords:
            coords = coords - coords.mean(axis=0)

        # 断言：确保原子类型映射正确
        assert all(0 <= idx < len(ATOM_TO_IDX) for idx in atom_indices), \
            "Invalid atom indices detected"

        return {
            'atom_types': torch.tensor(atom_indices, dtype=torch.long),
            'coords': torch.tensor(coords, dtype=torch.float32),
            'name': sample.get('name', f'sample_{idx}')
        }
