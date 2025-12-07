#!/usr/bin/env python3
"""
PDB结构处理和特征编码模块
"""

import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义
# =============================================================================

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

AA_LIST = list(AA_3TO1.keys())

# 氨基酸理化性质
AA_PROPERTIES = {
    'ALA': {'hydro': 1.8, 'volume': 88.6, 'charge': 0, 'polar': 0, 'aromatic': 0, 'pka': 0},
    'ARG': {'hydro': -4.5, 'volume': 173.4, 'charge': 1, 'polar': 1, 'aromatic': 0, 'pka': 12.5},
    'ASN': {'hydro': -3.5, 'volume': 114.1, 'charge': 0, 'polar': 1, 'aromatic': 0, 'pka': 0},
    'ASP': {'hydro': -3.5, 'volume': 111.1, 'charge': -1, 'polar': 1, 'aromatic': 0, 'pka': 3.9},
    'CYS': {'hydro': 2.5, 'volume': 108.5, 'charge': 0, 'polar': 0, 'aromatic': 0, 'pka': 8.3},
    'GLN': {'hydro': -3.5, 'volume': 143.8, 'charge': 0, 'polar': 1, 'aromatic': 0, 'pka': 0},
    'GLU': {'hydro': -3.5, 'volume': 138.4, 'charge': -1, 'polar': 1, 'aromatic': 0, 'pka': 4.1},
    'GLY': {'hydro': -0.4, 'volume': 60.1, 'charge': 0, 'polar': 0, 'aromatic': 0, 'pka': 0},
    'HIS': {'hydro': -3.2, 'volume': 153.2, 'charge': 0.5, 'polar': 1, 'aromatic': 1, 'pka': 6.0},
    'ILE': {'hydro': 4.5, 'volume': 166.7, 'charge': 0, 'polar': 0, 'aromatic': 0, 'pka': 0},
    'LEU': {'hydro': 3.8, 'volume': 166.7, 'charge': 0, 'polar': 0, 'aromatic': 0, 'pka': 0},
    'LYS': {'hydro': -3.9, 'volume': 168.6, 'charge': 1, 'polar': 1, 'aromatic': 0, 'pka': 10.5},
    'MET': {'hydro': 1.9, 'volume': 162.9, 'charge': 0, 'polar': 0, 'aromatic': 0, 'pka': 0},
    'PHE': {'hydro': 2.8, 'volume': 189.9, 'charge': 0, 'polar': 0, 'aromatic': 1, 'pka': 0},
    'PRO': {'hydro': -1.6, 'volume': 112.7, 'charge': 0, 'polar': 0, 'aromatic': 0, 'pka': 0},
    'SER': {'hydro': -0.8, 'volume': 89.0, 'charge': 0, 'polar': 1, 'aromatic': 0, 'pka': 0},
    'THR': {'hydro': -0.7, 'volume': 116.1, 'charge': 0, 'polar': 1, 'aromatic': 0, 'pka': 0},
    'TRP': {'hydro': -0.9, 'volume': 227.8, 'charge': 0, 'polar': 0, 'aromatic': 1, 'pka': 0},
    'TYR': {'hydro': -1.3, 'volume': 193.6, 'charge': 0, 'polar': 1, 'aromatic': 1, 'pka': 10.1},
    'VAL': {'hydro': 4.2, 'volume': 140.0, 'charge': 0, 'polar': 0, 'aromatic': 0, 'pka': 0}
}

# 催化残基先验概率
CATALYTIC_PRIOR = {
    'HIS': 0.25, 'ASP': 0.20, 'GLU': 0.18, 'CYS': 0.15, 'SER': 0.12,
    'LYS': 0.10, 'ARG': 0.08, 'TYR': 0.06, 'ASN': 0.04, 'THR': 0.03
}

# 催化角色映射
ROLE_MAPPING = {
    'nucleophile': 0, 'proton_donor': 1, 'proton_acceptor': 2,
    'electrostatic_stabilizer': 3, 'metal_binding': 4, 'covalent_catalyst': 5,
    'activator': 6, 'steric_role': 7, 'other': 8
}


# =============================================================================
# PDB处理器
# =============================================================================

class PDBProcessor:
    """PDB结构处理器"""

    def __init__(self, pdb_dir: str = "./data/pdb_structures"):
        self.pdb_dir = Path(pdb_dir)
        self.pdb_dir.mkdir(parents=True, exist_ok=True)

        try:
            from Bio.PDB import PDBParser
            self.parser = PDBParser(QUIET=True)
            self.biopython = True
        except ImportError:
            self.biopython = False
            logger.warning("BioPython未安装，使用简化解析器")

    def download_pdb(self, pdb_id: str) -> Optional[Path]:
        """下载PDB文件"""
        pdb_id = pdb_id.lower()
        pdb_file = self.pdb_dir / f"{pdb_id}.pdb"

        if pdb_file.exists():
            return pdb_file

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(pdb_file, 'w') as f:
                    f.write(r.text)
                return pdb_file
        except Exception as e:
            logger.warning(f"下载PDB {pdb_id} 失败: {e}")
        return None

    def parse_pdb(self, pdb_path: Path) -> Dict:
        """解析PDB文件"""
        if self.biopython:
            return self._parse_biopython(pdb_path)
        return self._parse_simple(pdb_path)

    def _parse_biopython(self, pdb_path: Path) -> Dict:
        """使用BioPython解析"""
        structure = self.parser.get_structure('protein', str(pdb_path))

        residues = []
        sequence = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    het, resseq, icode = residue.get_id()
                    if het != ' ':
                        continue

                    res_name = residue.get_resname()
                    if res_name not in AA_3TO1:
                        continue

                    atoms = {}
                    ca_coord = cb_coord = n_coord = c_coord = None

                    for atom in residue:
                        coord = atom.get_coord().tolist()
                        atoms[atom.get_name()] = coord
                        if atom.get_name() == 'CA':
                            ca_coord = coord
                        elif atom.get_name() == 'CB':
                            cb_coord = coord
                        elif atom.get_name() == 'N':
                            n_coord = coord
                        elif atom.get_name() == 'C':
                            c_coord = coord

                    if ca_coord:
                        residues.append({
                            'name': res_name,
                            'number': resseq,
                            'chain': chain.get_id(),
                            'ca_coord': ca_coord,
                            'cb_coord': cb_coord or ca_coord,
                            'n_coord': n_coord,
                            'c_coord': c_coord,
                            'atoms': atoms
                        })
                        sequence.append(AA_3TO1[res_name])

        return {
            'pdb_id': pdb_path.stem,
            'residues': residues,
            'sequence': ''.join(sequence),
            'num_residues': len(residues)
        }

    def _parse_simple(self, pdb_path: Path) -> Dict:
        """简化解析（无BioPython）"""
        residues = []
        sequence = []
        current = None

        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue

                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21]
                try:
                    res_num = int(line[22:26].strip())
                except ValueError:
                    continue
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])

                if res_name not in AA_3TO1:
                    continue

                key = (chain, res_num)
                if current is None or (current['chain'], current['number']) != key:
                    if current and current.get('ca_coord'):
                        residues.append(current)
                        sequence.append(AA_3TO1[current['name']])
                    current = {
                        'name': res_name, 'number': res_num, 'chain': chain,
                        'ca_coord': None, 'cb_coord': None,
                        'n_coord': None, 'c_coord': None, 'atoms': {}
                    }

                current['atoms'][atom_name] = [x, y, z]
                if atom_name == 'CA':
                    current['ca_coord'] = [x, y, z]
                elif atom_name == 'CB':
                    current['cb_coord'] = [x, y, z]
                elif atom_name == 'N':
                    current['n_coord'] = [x, y, z]
                elif atom_name == 'C':
                    current['c_coord'] = [x, y, z]

        if current and current.get('ca_coord'):
            residues.append(current)
            sequence.append(AA_3TO1[current['name']])

        # 填充缺失的CB坐标
        for r in residues:
            if not r['cb_coord']:
                r['cb_coord'] = r['ca_coord']

        return {
            'pdb_id': pdb_path.stem,
            'residues': residues,
            'sequence': ''.join(sequence),
            'num_residues': len(residues)
        }


# =============================================================================
# 特征编码器
# =============================================================================

class FeatureEncoder:
    """残基和结构特征编码器"""

    def __init__(self):
        self.aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

    def encode_residue(self, res_name: str) -> np.ndarray:
        """编码单个残基 -> 28维特征"""
        features = []

        # One-hot编码 (20维)
        one_hot = [0] * 20
        if res_name in self.aa_to_idx:
            one_hot[self.aa_to_idx[res_name]] = 1
        features.extend(one_hot)

        # 理化性质 (7维)
        props = AA_PROPERTIES.get(res_name, AA_PROPERTIES['ALA'])
        features.extend([
            props['hydro'] / 5.0,
            props['volume'] / 230.0,
            props['charge'],
            props['polar'],
            props['aromatic'],
            props['pka'] / 14.0 if props['pka'] > 0 else 0,
            CATALYTIC_PRIOR.get(res_name, 0.01)
        ])

        # 是否常见催化残基 (1维)
        features.append(1.0 if res_name in CATALYTIC_PRIOR else 0.0)

        return np.array(features, dtype=np.float32)

    def encode_structure(self, structure_data: Dict, cutoff: float = 10.0) -> Dict:
        """编码整个蛋白质结构"""
        residues = structure_data['residues']
        n = len(residues)

        # 节点特征
        node_features = np.zeros((n, 28), dtype=np.float32)
        ca_coords = np.zeros((n, 3), dtype=np.float32)
        cb_coords = np.zeros((n, 3), dtype=np.float32)
        residue_info = []

        for i, res in enumerate(residues):
            node_features[i] = self.encode_residue(res['name'])
            ca_coords[i] = res['ca_coord']
            cb_coords[i] = res.get('cb_coord', res['ca_coord'])
            residue_info.append((res['chain'], res['number'], res['name']))

        # 构建边和边特征
        edge_index, edge_attr = self._build_edges(ca_coords, cb_coords, cutoff)

        # 空间特征
        spatial_features = self._compute_spatial_features(ca_coords)
        node_features = np.concatenate([node_features, spatial_features], axis=1)

        return {
            'node_features': node_features,
            'ca_coords': ca_coords,
            'cb_coords': cb_coords,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'residue_info': residue_info,
            'sequence': structure_data.get('sequence', '')
        }

    def _build_edges(self, ca_coords: np.ndarray, cb_coords: np.ndarray,
                     cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
        """构建带几何特征的边"""
        n = len(ca_coords)

        # 计算CA距离矩阵
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # 找边
        src, dst = np.where((dist_matrix < cutoff) & (dist_matrix > 0))

        edge_attr = []
        for s, d in zip(src, dst):
            dist = dist_matrix[s, d]
            direction = (ca_coords[d] - ca_coords[s]) / (dist + 1e-8)
            cb_dist = np.linalg.norm(cb_coords[d] - cb_coords[s])
            seq_dist = abs(d - s)

            edge_attr.append([
                dist / cutoff,                    # 归一化CA距离
                cb_dist / cutoff,                 # 归一化CB距离
                1.0 / (dist + 1.0),              # 距离倒数
                np.exp(-dist**2 / 32),           # RBF编码
                min(seq_dist, 20) / 20.0,        # 归一化序列距离
                direction[0], direction[1], direction[2]  # 方向向量
            ])

        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, 8), dtype=np.float32)

        return edge_index, edge_attr

    def _compute_spatial_features(self, coords: np.ndarray) -> np.ndarray:
        """计算空间环境特征 (5维)"""
        n = len(coords)
        features = np.zeros((n, 5), dtype=np.float32)

        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        centroid = coords.mean(axis=0)

        for i in range(n):
            # 局部密度 (8Å, 12Å)
            features[i, 0] = np.sum(dist_matrix[i] < 8.0) / n
            features[i, 1] = np.sum(dist_matrix[i] < 12.0) / n

            # 平均邻居距离
            sorted_dists = np.sort(dist_matrix[i])
            features[i, 2] = np.mean(sorted_dists[1:min(11, n)]) / 20.0 if n > 1 else 0

            # 到质心距离 (埋藏度)
            depth = np.linalg.norm(coords[i] - centroid)
            max_depth = np.max(np.linalg.norm(coords - centroid, axis=1))
            features[i, 3] = depth / (max_depth + 1e-8)

            # 局部曲率估计
            if n > 10:
                neighbors = np.argsort(dist_matrix[i])[1:11]
                neighbor_coords = coords[neighbors]
                local_centroid = neighbor_coords.mean(axis=0)
                features[i, 4] = np.linalg.norm(coords[i] - local_centroid) / 5.0

        return features
