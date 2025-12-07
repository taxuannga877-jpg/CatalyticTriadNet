#!/usr/bin/env python3
"""
纳米酶数据集模块
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List
import numpy as np
from .constraints import CatalyticConstraints
class NanozymeDataset(Dataset):
    """纳米酶/催化分子数据集"""
    def __init__(self, data_dir: str, max_atoms: int = 100):
        self.data_dir = Path(data_dir)
        self.max_atoms = max_atoms
        self.samples = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        samples = []
        # 支持多种格式
        for ext in ['*.xyz', '*.mol', '*.json']:
            for f in self.data_dir.glob(ext):
                try:
                    sample = self._parse_file(f)
                    if sample and len(sample['atom_types']) <= self.max_atoms:
                        samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to parse {f}: {e}")
        return samples
    
    def _parse_file(self, filepath: Path) -> Optional[Dict]:
        """解析分子文件"""
        if filepath.suffix == '.xyz':
            return self._parse_xyz(filepath)
        elif filepath.suffix == '.json':
            with open(filepath) as f:
                return json.load(f)
        return None
    
    def _parse_xyz(self, filepath: Path) -> Dict:
        """解析XYZ文件"""
        with open(filepath) as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        atoms = []
        coords = []
        
        for line in lines[2:2 + n_atoms]:
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return {
            'atom_types': atoms,
            'coords': coords,
            'name': filepath.stem
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 转换原子类型到索引
        atom_indices = [ATOM_TO_IDX.get(a, 0) for a in sample['atom_types']]
        coords = np.array(sample['coords'], dtype=np.float32)
        
        # 中心化坐标
        coords = coords - coords.mean(axis=0)
        
        return {
            'atom_types': torch.tensor(atom_indices, dtype=torch.long),
            'coords': torch.tensor(coords, dtype=torch.float32)
        }
