#!/usr/bin/env python3
"""
催化位点分析模块：三联体检测、金属中心分析、氢键网络
"""

import numpy as np
from typing import Dict, List, Optional, Set
from collections import defaultdict
from itertools import product
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义
# =============================================================================

# 经典催化三联体模式 (参考M-CSA数据库)
TRIAD_PATTERNS = {
    'serine_protease': {
        'residues': [('SER', 'nucleophile'), ('HIS', 'general_base'), ('ASP', 'electrostatic')],
        'distances': {'SER-HIS': (2.5, 4.0), 'HIS-ASP': (2.5, 4.0), 'SER-ASP': (6.0, 10.0)},
        'ec_class': [3, 4],
    },
    'cysteine_protease': {
        'residues': [('CYS', 'nucleophile'), ('HIS', 'general_base'), ('ASN', 'electrostatic')],
        'distances': {'CYS-HIS': (3.0, 4.5), 'HIS-ASN': (2.5, 4.0), 'CYS-ASN': (6.0, 10.0)},
        'ec_class': [3],
    },
    'cysteine_protease_asp': {
        'residues': [('CYS', 'nucleophile'), ('HIS', 'general_base'), ('ASP', 'electrostatic')],
        'distances': {'CYS-HIS': (3.0, 4.5), 'HIS-ASP': (2.5, 4.0), 'CYS-ASP': (6.0, 10.0)},
        'ec_class': [3],
    },
    'threonine_protease': {
        'residues': [('THR', 'nucleophile'), ('LYS', 'general_base'), ('ASP', 'electrostatic')],
        'distances': {'THR-LYS': (3.0, 5.0), 'LYS-ASP': (3.0, 5.0), 'THR-ASP': (5.0, 9.0)},
        'ec_class': [3],
    },
    'aspartic_protease': {
        'residues': [('ASP', 'nucleophile'), ('ASP', 'general_acid'), ('THR', 'stabilizer')],
        'distances': {'ASP-ASP': (2.5, 4.0), 'ASP-THR': (3.0, 5.0)},
        'ec_class': [3],
    },
}

# 金属离子配位几何
METAL_COORDINATION = {
    'ZN': {'coord_number': [4, 5, 6], 'geometry': ['tetrahedral', 'trigonal_bipyramidal', 'octahedral'],
           'common_ligands': ['HIS', 'CYS', 'ASP', 'GLU'], 'ideal_distance': 2.0},
    'MG': {'coord_number': [6], 'geometry': ['octahedral'],
           'common_ligands': ['ASP', 'GLU', 'SER', 'THR'], 'ideal_distance': 2.1},
    'MN': {'coord_number': [6], 'geometry': ['octahedral'],
           'common_ligands': ['HIS', 'ASP', 'GLU'], 'ideal_distance': 2.2},
    'FE': {'coord_number': [4, 5, 6], 'geometry': ['tetrahedral', 'square_pyramidal', 'octahedral'],
           'common_ligands': ['HIS', 'CYS', 'TYR'], 'ideal_distance': 2.0},
    'CU': {'coord_number': [4, 5], 'geometry': ['square_planar', 'trigonal_bipyramidal'],
           'common_ligands': ['HIS', 'CYS', 'MET'], 'ideal_distance': 2.0},
    'CA': {'coord_number': [6, 7, 8], 'geometry': ['octahedral', 'pentagonal_bipyramidal'],
           'common_ligands': ['ASP', 'GLU', 'ASN'], 'ideal_distance': 2.4},
}

# 双金属中心模式
BIMETALLIC_PATTERNS = {
    'phosphodiesterase': {
        'metals': ['MG', 'MG'], 'distance_range': (3.4, 4.2),
        'bridging_ligands': ['ASP', 'GLU', 'HOH'],
        'ec_class': [3, 1],
    },
    'purple_acid_phosphatase': {
        'metals': ['FE', 'ZN'], 'distance_range': (3.0, 3.5),
        'bridging_ligands': ['ASP', 'HOH'],
        'ec_class': [3],
    },
    'urease': {
        'metals': ['NI', 'NI'], 'distance_range': (3.5, 3.7),
        'bridging_ligands': ['LYS', 'HOH'],
        'ec_class': [3],
    },
    'metallo_beta_lactamase': {
        'metals': ['ZN', 'ZN'], 'distance_range': (3.4, 4.5),
        'bridging_ligands': ['ASP', 'HIS', 'HOH'],
        'ec_class': [3],
    },
}

METAL_NAMES = {"ZN", "MG", "MN", "FE", "CU", "CO", "NI", "CA", "NA", "K", "CD", "MO", "W", "V",
               "FE2", "FE3", "ZN2", "MG2", "MN2", "CU2", "CO2", "NI2", "CA2"}


# =============================================================================
# 氢键网络分析
# =============================================================================

class HydrogenBondAnalyzer:
    """氢键网络分析"""

    DONORS = {'N', 'NZ', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'OG', 'OG1', 'OH', 'SG'}
    ACCEPTORS = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'ND1', 'NE2', 'SD'}

    def __init__(self, distance_cutoff: float = 3.5, angle_cutoff: float = 120.0):
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff

    def find_hbonds(self, residues: List[Dict]) -> List[Dict]:
        """查找氢键"""
        hbonds = []

        for i, res_i in enumerate(residues):
            atoms_i = res_i.get('atoms', {})

            for j, res_j in enumerate(residues):
                if abs(i - j) < 2:
                    continue

                atoms_j = res_j.get('atoms', {})

                for donor_name in self.DONORS:
                    if donor_name not in atoms_i:
                        continue
                    donor_coord = np.array(atoms_i[donor_name])

                    for acceptor_name in self.ACCEPTORS:
                        if acceptor_name not in atoms_j:
                            continue
                        acceptor_coord = np.array(atoms_j[acceptor_name])

                        dist = np.linalg.norm(donor_coord - acceptor_coord)

                        if dist < self.distance_cutoff:
                            hbonds.append({
                                'donor_residue': i,
                                'acceptor_residue': j,
                                'donor_atom': donor_name,
                                'acceptor_atom': acceptor_name,
                                'distance': float(dist),
                            })

        return hbonds

    def compute_hbond_features(self, residue_idx: int, hbonds: List[Dict],
                               n_residues: int) -> np.ndarray:
        """计算氢键相关特征 (3维)"""
        n_donor = sum(1 for hb in hbonds if hb['donor_residue'] == residue_idx)
        n_acceptor = sum(1 for hb in hbonds if hb['acceptor_residue'] == residue_idx)

        centrality = (n_donor + n_acceptor) / max(len(hbonds), 1)

        return np.array([
            n_donor / 5.0,
            n_acceptor / 5.0,
            centrality
        ], dtype=np.float32)


# =============================================================================
# 催化三联体检测器
# =============================================================================

class TriadDetector:
    """智能催化三联体检测器"""

    def __init__(self):
        self.patterns = TRIAD_PATTERNS

    def detect_triads(self, residues: List[Dict], coords: np.ndarray,
                      catalytic_residues: List[Dict],
                      predicted_ec1: int = None) -> List[Dict]:
        """检测催化三联体"""
        triads = []

        for pattern_name, pattern in self.patterns.items():
            # EC类别过滤
            if predicted_ec1 and predicted_ec1 not in pattern.get('ec_class', []):
                continue

            matched = self._match_pattern(
                pattern, residues, coords, catalytic_residues
            )
            triads.extend(matched)

        # 去重和排序
        triads = self._deduplicate_triads(triads)
        triads.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return triads

    def _match_pattern(self, pattern: Dict, residues: List[Dict],
                       coords: np.ndarray, catalytic_residues: List[Dict]) -> List[Dict]:
        """匹配单个三联体模式"""
        matches = []
        required = pattern['residues']
        distances = pattern.get('distances', {})

        # 创建残基名到索引的映射
        cat_by_name = defaultdict(list)
        for cat_res in catalytic_residues:
            cat_by_name[cat_res['resname']].append(cat_res)

        # 尝试所有组合
        candidates = [[], [], []]
        for i, (req_name, req_role) in enumerate(required):
            for cat_res in cat_by_name.get(req_name, []):
                candidates[i].append(cat_res)

        # 检查所有组合
        for combo in product(*candidates):
            if len(set(c['index'] for c in combo)) != 3:
                continue

            # 检查距离约束
            valid = True
            actual_distances = {}

            idx0, idx1, idx2 = combo[0]['index'], combo[1]['index'], combo[2]['index']
            c0, c1, c2 = coords[idx0], coords[idx1], coords[idx2]

            d01 = np.linalg.norm(c0 - c1)
            d12 = np.linalg.norm(c1 - c2)
            d02 = np.linalg.norm(c0 - c2)

            for (r1, _), (r2, _), d in [(required[0], required[1], d01),
                                         (required[1], required[2], d12),
                                         (required[0], required[2], d02)]:
                key = f"{r1}-{r2}"
                alt_key = f"{r2}-{r1}"

                if key in distances:
                    d_min, d_max = distances[key]
                    if not (d_min <= d <= d_max):
                        valid = False
                        break
                elif alt_key in distances:
                    d_min, d_max = distances[alt_key]
                    if not (d_min <= d <= d_max):
                        valid = False
                        break

                actual_distances[key] = d

            if valid:
                avg_prob = np.mean([c['site_prob'] for c in combo])
                confidence = avg_prob * 0.7 + 0.3

                matches.append({
                    'pattern': pattern_name,
                    'residues': list(combo),
                    'distances': actual_distances,
                    'confidence': float(confidence),
                })

        return matches

    def _deduplicate_triads(self, triads: List[Dict]) -> List[Dict]:
        """去重"""
        seen = set()
        unique = []

        for t in triads:
            key = tuple(sorted([r['index'] for r in t['residues']]))
            if key not in seen:
                seen.add(key)
                unique.append(t)

        return unique


# =============================================================================
# 双金属中心检测器
# =============================================================================

class BimetallicCenterDetector:
    """双金属活性中心检测器"""

    def __init__(self):
        self.bimetallic_patterns = BIMETALLIC_PATTERNS
        self.metal_coordination = METAL_COORDINATION

    def detect_bimetallic_centers(self, metals: List[Dict], residues: List[Dict],
                                   coords: np.ndarray) -> List[Dict]:
        """检测双金属中心"""
        if len(metals) < 2:
            return []

        bimetallic_centers = []
        metal_coords = np.array([m['coord'] for m in metals])

        for i in range(len(metals)):
            for j in range(i + 1, len(metals)):
                m1, m2 = metals[i], metals[j]
                m1_name = m1['name'].upper().rstrip('0123456789')
                m2_name = m2['name'].upper().rstrip('0123456789')

                dist = np.linalg.norm(metal_coords[i] - metal_coords[j])

                # 匹配双金属模式
                matched_pattern = None
                for pattern_name, pattern in self.bimetallic_patterns.items():
                    req_metals = set(pattern['metals'])
                    actual_metals = {m1_name, m2_name}

                    if req_metals == actual_metals or (len(req_metals) == 1 and m1_name == m2_name):
                        d_min, d_max = pattern['distance_range']
                        if d_min <= dist <= d_max:
                            matched_pattern = pattern_name
                            break

                # 查找桥连配体
                bridging_residues = self._find_bridging_ligands(
                    metal_coords[i], metal_coords[j], residues, coords
                )

                if matched_pattern or (3.0 <= dist <= 5.0 and bridging_residues):
                    bimetallic_centers.append({
                        'metal1': m1,
                        'metal2': m2,
                        'distance': float(dist),
                        'pattern': matched_pattern,
                        'bridging_residues': bridging_residues,
                        'midpoint': (metal_coords[i] + metal_coords[j]) / 2,
                    })

        return bimetallic_centers

    def _find_bridging_ligands(self, coord1: np.ndarray, coord2: np.ndarray,
                                residues: List[Dict], coords: np.ndarray,
                                cutoff: float = 3.0) -> List[Dict]:
        """查找桥连配体"""
        bridging = []

        for i, res in enumerate(residues):
            ca = coords[i]
            d1 = np.linalg.norm(ca - coord1)
            d2 = np.linalg.norm(ca - coord2)

            if d1 < cutoff and d2 < cutoff:
                bridging.append({
                    'index': i,
                    'resname': res['name'],
                    'resseq': res['number'],
                    'chain': res['chain'],
                    'dist_to_m1': float(d1),
                    'dist_to_m2': float(d2),
                })

        return bridging

    def analyze_coordination_geometry(self, metal: Dict, residues: List[Dict],
                                       coords: np.ndarray, cutoff: float = 3.0) -> Dict:
        """分析单金属配位几何"""
        metal_coord = metal['coord']
        metal_name = metal['name'].upper().rstrip('0123456789')

        distances = np.linalg.norm(coords - metal_coord, axis=1)
        ligand_indices = np.where(distances < cutoff)[0]

        ligand_info = []
        for idx in ligand_indices:
            ligand_info.append({
                'index': int(idx),
                'resname': residues[idx]['name'],
                'distance': float(distances[idx]),
            })

        coord_number = len(ligand_indices)
        expected = self.metal_coordination.get(metal_name, {})

        geometry = 'unknown'
        if coord_number == 4:
            geometry = 'tetrahedral'
        elif coord_number == 5:
            geometry = 'trigonal_bipyramidal'
        elif coord_number == 6:
            geometry = 'octahedral'

        return {
            'metal': metal,
            'coordination_number': coord_number,
            'geometry': geometry,
            'ligands': ligand_info,
            'expected_coord': expected.get('coord_number', []),
            'ideal_distance': expected.get('ideal_distance', 2.0),
        }
