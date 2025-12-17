#!/usr/bin/env python3
"""
底物定义和NAC条件
支持6种经典纳米酶底物：TMB, pNPP, ABTS, OPD, H₂O₂, GSH
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SubstrateDefinition:
    """底物定义"""
    name: str
    full_name: str
    enzyme_type: str  # 'peroxidase', 'phosphatase', 'catalase', 'GPx'
    smiles: str  # SMILES结构
    detection_wavelength: Optional[int]  # 检测波长 (nm)

    # NAC几何条件
    nac_conditions: Dict

    # 所需催化中心类型
    required_catalytic_features: Dict

    # 反应机制描述
    mechanism: str

    # 文献使用频率
    usage_frequency: int  # 1-5星


# =============================================================================
# 6种经典纳米酶底物定义
# =============================================================================

SUBSTRATE_LIBRARY = {
    # 1. TMB - 最常用的过氧化物酶底物
    'TMB': SubstrateDefinition(
        name='TMB',
        full_name='3,3\',5,5\'-Tetramethylbenzidine',
        enzyme_type='peroxidase',
        smiles='CC1=C(C=C(C=C1N)C)N',
        detection_wavelength=652,

        nac_conditions={
            # 金属中心到底物距离
            'metal_substrate_distance': {
                'range': (2.0, 2.8),
                'optimal': 2.3,
                'tolerance': 0.3,
                'weight': 0.3
            },
            # H₂O₂结合位点距离
            'H2O2_binding_distance': {
                'range': (2.5, 3.5),
                'optimal': 3.0,
                'tolerance': 0.5,
                'weight': 0.25
            },
            # 电子转移距离
            'electron_transfer_distance': {
                'range': (3.0, 4.5),
                'optimal': 3.5,
                'tolerance': 0.5,
                'weight': 0.25
            },
            # 氧化位点可及性
            'oxidation_site_accessibility': {
                'min_clearance': 3.0,  # Å
                'weight': 0.2
            }
        },

        required_catalytic_features={
            'metal_center': ['Fe', 'Cu', 'Mn', 'Co', 'Ce'],  # 金属类型
            'oxidation_state': [2, 3],  # 氧化态
            'coordination_number': [4, 5, 6],
            'alternative': {  # 或者非金属氧化还原中心
                'redox_residues': ['CYS', 'TYR', 'TRP']
            }
        },

        mechanism='H₂O₂ + TMB (还原态) → H₂O + TMB (氧化态, 蓝色)',
        usage_frequency=5
    ),

    # 2. pNPP - 磷酸酶金标准
    'pNPP': SubstrateDefinition(
        name='pNPP',
        full_name='p-Nitrophenyl phosphate',
        enzyme_type='phosphatase',
        smiles='C1=CC(=CC=C1[N+](=O)[O-])OP(=O)(O)O',
        detection_wavelength=405,

        nac_conditions={
            # 亲核试剂到磷原子距离
            'nucleophile_P_distance': {
                'range': (2.7, 3.3),
                'optimal': 3.0,
                'tolerance': 0.3,
                'weight': 0.35
            },
            # 广义碱到亲核试剂距离
            'base_nucleophile_distance': {
                'range': (3.0, 4.5),
                'optimal': 3.5,
                'tolerance': 0.5,
                'weight': 0.25
            },
            # 攻击角度 (O-P-O)
            'attack_angle': {
                'range': (160, 180),
                'optimal': 170,
                'tolerance': 10,
                'weight': 0.2
            },
            # 金属离子到磷原子距离（如果有金属）
            'metal_P_distance': {
                'range': (3.5, 4.5),
                'optimal': 4.0,
                'tolerance': 0.5,
                'weight': 0.2
            }
        },

        required_catalytic_features={
            'nucleophile': ['SER', 'CYS', 'THR'],  # 亲核试剂
            'general_base': ['HIS', 'LYS'],  # 广义碱
            'electrostatic': ['ASP', 'GLU'],  # 静电稳定
            'metal_center': ['Zn', 'Mg', 'Mn', 'Fe'],  # 可选金属
        },

        mechanism='pNPP + H₂O → p-nitrophenol (黄色) + 磷酸',
        usage_frequency=4
    ),

    # 3. ABTS - 水溶性过氧化物酶底物
    'ABTS': SubstrateDefinition(
        name='ABTS',
        full_name='2,2\'-Azino-bis(3-ethylbenzothiazoline-6-sulfonic acid)',
        enzyme_type='peroxidase',
        smiles='CCN1C2=CC(=C(C=C2SC1=NC3=CC=C(C=C3)S(=O)(=O)O)S(=O)(=O)O)S(=O)(=O)O',
        detection_wavelength=414,

        nac_conditions={
            # 金属中心到底物距离
            'metal_substrate_distance': {
                'range': (2.0, 2.8),
                'optimal': 2.4,
                'tolerance': 0.3,
                'weight': 0.3
            },
            # 氧化位点距离
            'oxidation_site_distance': {
                'range': (3.0, 4.5),
                'optimal': 3.5,
                'tolerance': 0.5,
                'weight': 0.25
            },
            # H₂O₂配位
            'H2O2_coordination': {
                'required': True,
                'distance': (2.5, 3.5),
                'weight': 0.25
            },
            # 底物可及性（ABTS较大）
            'substrate_accessibility': {
                'min_pocket_size': 8.0,  # Å
                'weight': 0.2
            }
        },

        required_catalytic_features={
            'metal_center': ['Fe', 'Cu', 'Mn', 'Co'],
            'oxidation_state': [2, 3],
            'coordination_number': [4, 5, 6],
        },

        mechanism='H₂O₂ + ABTS (还原态) → H₂O + ABTS•⁺ (绿色)',
        usage_frequency=4
    ),

    # 4. OPD - 高灵敏度过氧化物酶底物
    'OPD': SubstrateDefinition(
        name='OPD',
        full_name='o-Phenylenediamine',
        enzyme_type='peroxidase',
        smiles='C1=CC=C(C(=C1)N)N',
        detection_wavelength=450,

        nac_conditions={
            # 金属中心到底物距离
            'metal_substrate_distance': {
                'range': (2.0, 2.8),
                'optimal': 2.3,
                'tolerance': 0.3,
                'weight': 0.35
            },
            # 氨基氧化位点距离
            'amine_oxidation_distance': {
                'range': (3.0, 4.0),
                'optimal': 3.5,
                'tolerance': 0.5,
                'weight': 0.3
            },
            # H₂O₂结合
            'H2O2_binding_distance': {
                'range': (2.5, 3.5),
                'optimal': 3.0,
                'tolerance': 0.5,
                'weight': 0.25
            },
            # 双氨基间距（OPD特征）
            'diamine_spacing': {
                'range': (2.5, 3.0),
                'optimal': 2.7,
                'tolerance': 0.3,
                'weight': 0.1
            }
        },

        required_catalytic_features={
            'metal_center': ['Fe', 'Cu', 'Mn', 'Co'],
            'oxidation_state': [2, 3],
            'coordination_number': [4, 5, 6],
        },

        mechanism='H₂O₂ + OPD → H₂O + 2,3-diaminophenazine (橙黄色)',
        usage_frequency=3
    ),

    # 5. H₂O₂ - 过氧化氢（直接底物/产物）
    'H2O2': SubstrateDefinition(
        name='H2O2',
        full_name='Hydrogen peroxide',
        enzyme_type='catalase',
        smiles='OO',
        detection_wavelength=240,  # UV检测

        nac_conditions={
            # 金属中心到H₂O₂距离
            'metal_H2O2_distance': {
                'range': (2.0, 2.5),
                'optimal': 2.2,
                'tolerance': 0.3,
                'weight': 0.4
            },
            # O-O键活化距离
            'OO_activation_distance': {
                'range': (1.4, 1.6),  # O-O键长
                'optimal': 1.5,
                'tolerance': 0.1,
                'weight': 0.3
            },
            # 质子转移距离
            'proton_transfer_distance': {
                'range': (2.5, 3.5),
                'optimal': 3.0,
                'tolerance': 0.5,
                'weight': 0.3
            }
        },

        required_catalytic_features={
            'metal_center': ['Fe', 'Mn', 'Cu'],  # 过氧化氢酶金属
            'oxidation_state': [2, 3],
            'coordination_number': [5, 6],
            'alternative': {
                'redox_residues': ['CYS', 'TYR']
            }
        },

        mechanism='2 H₂O₂ → 2 H₂O + O₂ (catalase) 或 H₂O₂ + 底物 → H₂O + 氧化产物 (peroxidase)',
        usage_frequency=3
    ),

    # 6. GSH - 谷胱甘肽（GPx活性）
    'GSH': SubstrateDefinition(
        name='GSH',
        full_name='Glutathione (reduced)',
        enzyme_type='GPx',
        smiles='C(CC(=O)NC(CS)C(=O)NCC(=O)O)C(C(=O)O)N',
        detection_wavelength=412,  # DTNB法

        nac_conditions={
            # 巯基到活性中心距离
            'thiol_active_site_distance': {
                'range': (3.0, 4.0),
                'optimal': 3.5,
                'tolerance': 0.5,
                'weight': 0.35
            },
            # H₂O₂结合距离
            'H2O2_binding_distance': {
                'range': (2.5, 3.5),
                'optimal': 3.0,
                'tolerance': 0.5,
                'weight': 0.3
            },
            # 二硫键形成距离
            'disulfide_formation_distance': {
                'range': (2.0, 2.5),
                'optimal': 2.2,
                'tolerance': 0.3,
                'weight': 0.25
            },
            # 底物口袋大小（GSH较大）
            'pocket_size': {
                'min_size': 10.0,  # Å
                'weight': 0.1
            }
        },

        required_catalytic_features={
            'metal_center': ['Se', 'Fe', 'Cu'],  # GPx通常含硒
            'redox_residues': ['CYS', 'SEC'],  # 半胱氨酸或硒代半胱氨酸
            'coordination_number': [4, 5, 6],
        },

        mechanism='2 GSH + H₂O₂ → GSSG + 2 H₂O',
        usage_frequency=3
    ),
}


# =============================================================================
# 底物分类和查询函数
# =============================================================================

def get_substrate_by_enzyme_type(enzyme_type: str) -> List[str]:
    """根据酶类型获取底物列表"""
    return [
        name for name, substrate in SUBSTRATE_LIBRARY.items()
        if substrate.enzyme_type == enzyme_type
    ]


def get_peroxidase_substrates() -> List[str]:
    """获取所有过氧化物酶底物"""
    return get_substrate_by_enzyme_type('peroxidase')


def get_phosphatase_substrates() -> List[str]:
    """获取所有磷酸酶底物"""
    return get_substrate_by_enzyme_type('phosphatase')


def get_most_popular_substrates(top_n: int = 3) -> List[str]:
    """获取最常用的底物"""
    sorted_substrates = sorted(
        SUBSTRATE_LIBRARY.items(),
        key=lambda x: x[1].usage_frequency,
        reverse=True
    )
    return [name for name, _ in sorted_substrates[:top_n]]


# =============================================================================
# 底物兼容性规则
# =============================================================================

SUBSTRATE_COMPATIBILITY_RULES = {
    'TMB': {
        'required_functional_groups': ['metal_center', 'redox_site'],
        'incompatible_with': [],
        'optimal_pH': (3.0, 5.0),
        'requires_H2O2': True
    },
    'pNPP': {
        'required_functional_groups': ['nucleophile', 'general_base'],
        'incompatible_with': [],
        'optimal_pH': (8.0, 10.0),
        'requires_H2O2': False
    },
    'ABTS': {
        'required_functional_groups': ['metal_center', 'redox_site'],
        'incompatible_with': [],
        'optimal_pH': (4.0, 5.0),
        'requires_H2O2': True
    },
    'OPD': {
        'required_functional_groups': ['metal_center', 'redox_site'],
        'incompatible_with': [],
        'optimal_pH': (4.0, 6.0),
        'requires_H2O2': True
    },
    'H2O2': {
        'required_functional_groups': ['metal_center', 'catalase_site'],
        'incompatible_with': [],
        'optimal_pH': (6.0, 8.0),
        'requires_H2O2': False  # H₂O₂本身就是底物
    },
    'GSH': {
        'required_functional_groups': ['redox_site', 'thiol_binding'],
        'incompatible_with': [],
        'optimal_pH': (7.0, 8.0),
        'requires_H2O2': True
    },
}


# =============================================================================
# 辅助函数
# =============================================================================

def print_substrate_info(substrate_name: str):
    """打印底物详细信息"""
    if substrate_name not in SUBSTRATE_LIBRARY:
        print(f"未知底物: {substrate_name}")
        return

    substrate = SUBSTRATE_LIBRARY[substrate_name]

    print(f"\n{'='*60}")
    print(f"底物: {substrate.name} ({substrate.full_name})")
    print(f"{'='*60}")
    print(f"酶类型: {substrate.enzyme_type}")
    print(f"检测波长: {substrate.detection_wavelength} nm")
    print(f"使用频率: {'⭐' * substrate.usage_frequency}")
    print(f"\n反应机制:")
    print(f"  {substrate.mechanism}")
    print(f"\n所需催化特征:")
    for key, value in substrate.required_catalytic_features.items():
        print(f"  {key}: {value}")
    print(f"\nNAC条件:")
    for key, value in substrate.nac_conditions.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")


def get_all_substrate_names() -> List[str]:
    """获取所有支持的底物名称"""
    return list(SUBSTRATE_LIBRARY.keys())


def validate_substrate(substrate_name: str) -> bool:
    """验证底物是否支持"""
    return substrate_name in SUBSTRATE_LIBRARY


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'SubstrateDefinition',
    'SUBSTRATE_LIBRARY',
    'get_substrate_by_enzyme_type',
    'get_peroxidase_substrates',
    'get_phosphatase_substrates',
    'get_most_popular_substrates',
    'SUBSTRATE_COMPATIBILITY_RULES',
    'print_substrate_info',
    'get_all_substrate_names',
    'validate_substrate',
]


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("支持的底物:")
    for name in get_all_substrate_names():
        print(f"  - {name}")

    print("\n最常用的3种底物:")
    for name in get_most_popular_substrates(3):
        print(f"  - {name}")

    print("\n过氧化物酶底物:")
    for name in get_peroxidase_substrates():
        print(f"  - {name}")

    # 打印TMB详细信息
    print_substrate_info('TMB')
    print_substrate_info('pNPP')
