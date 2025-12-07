"""
纳米酶可视化完整模块 v2.0
=====================================
适配: catalytic-triad-predictor-enhanced.py + 扩散模型生成结构

功能:
├── 2D可视化 (分子图、三联体、金属中心、完整报告)
├── 3D可视化 (空间分布、配位多面体、交互式、动画)
├── 专业软件导出 (PyMOL, ChimeraX, VMD)
└── 扩散模型适配 (RFdiffusion, ProteinMPNN, 自定义图数据)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ============================================================================
# 颜色配置
# ============================================================================
AA_COLORS = {
    'SER': '#FF6B6B', 'CYS': '#FF8E8E', 'THR': '#FFA5A5',  # 亲核
    'HIS': '#4ECDC4', 'LYS': '#45B7AA', 'ARG': '#3DAA9E',  # 碱性
    'ASP': '#FFE66D', 'GLU': '#FFD93D',                    # 酸性
    'PHE': '#C9B1FF', 'TYR': '#B8A0FF', 'TRP': '#A78BFA',  # 芳香
    'ALA': '#E8E8E8', 'VAL': '#E0E0E0', 'LEU': '#D8D8D8', 'ILE': '#D0D0D0',
    'MET': '#C8C8C8', 'PRO': '#C0C0C0', 'GLY': '#F0F0F0',
    'ASN': '#A8E6CF', 'GLN': '#98D9BE', 'UNK': '#CCCCCC'
}

METAL_COLORS = {
    'ZN': '#5C7AEA', 'MG': '#06D6A0', 'FE': '#EF476F', 'MN': '#9B5DE5',
    'CU': '#F4A261', 'CA': '#2EC4B6', 'NI': '#8338EC', 'CO': '#FB5607',
    'ZR': '#00B4D8', 'CE': '#90BE6D', 'DEFAULT': '#6C757D'
}

ROLE_COLORS = {
    'nucleophile': '#FF6B6B', 'general_base': '#4ECDC4', 'general_acid': '#FFE66D',
    'electrostatic': '#FFD93D', 'metal_ligand': '#5C7AEA',
    'transition_state_stabilizer': '#9B5DE5', 'non_catalytic': '#CCCCCC', 'other': '#888888'
}

# 配位几何模板
COORD_GEOMETRIES = {
    'tetrahedral': {'v': np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])/np.sqrt(3),
                    'f': [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]},
    'octahedral': {'v': np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]),
                   'f': [[0,2,4],[0,4,3],[0,3,5],[0,5,2],[1,2,4],[1,4,3],[1,3,5],[1,5,2]]},
    'square_planar': {'v': np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]])/np.sqrt(2),
                      'f': [[0,1,2,3]]},
    'trigonal_bipyramidal': {'v': np.array([[0,0,1],[0,0,-1],[1,0,0],[-0.5,0.866,0],[-0.5,-0.866,0]]),
                              'f': [[0,2,3],[0,3,4],[0,4,2],[1,2,3],[1,3,4],[1,4,2]]}
}


# ============================================================================
# 扩散模型适配器
# ============================================================================
class DiffusionModelAdapter:
    """
    扩散模型输出适配器
    支持: RFdiffusion, ProteinMPNN, 自定义图数据, PyG Data
    """
    
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'UNK']
    BOND_TYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    
    @staticmethod
    def from_rfdiffusion(output_path: str) -> Dict:
        """解析RFdiffusion输出"""
        with open(output_path) as f:
            data = json.load(f)
        
        return {
            'pdb_id': data.get('name', 'RFdiff_output'),
            'coords': np.array(data.get('coords', [])),
            'sequence': data.get('sequence', ''),
            'catalytic_residues': [
                {'index': int(h[1:])-1, 'resseq': int(h[1:]), 'chain': h[0],
                 'resname': 'UNK', 'site_prob': 1.0, 'role_name': 'other'}
                for h in data.get('hotspot_residues', [])
            ],
            'triads': [], 'metals': [], 'metal_centers': [], 'bimetallic_centers': []
        }
    
    @staticmethod
    def from_proteinmpnn(output_path: str) -> Dict:
        """解析ProteinMPNN输出"""
        with open(output_path) as f:
            data = json.load(f)
        
        fixed = data.get('fixed_positions', {})
        cat_res = []
        for chain, positions in fixed.items():
            for pos in positions:
                cat_res.append({
                    'index': pos-1, 'resseq': pos, 'chain': chain,
                    'resname': 'UNK', 'site_prob': 1.0, 'role_name': 'other'
                })
        
        return {
            'pdb_id': data.get('pdb_id', 'MPNN_output'),
            'catalytic_residues': cat_res,
            'triads': [], 'metals': [], 'metal_centers': [], 'bimetallic_centers': []
        }
    
    @staticmethod
    def from_graph_data(node_types: np.ndarray, edge_index: np.ndarray,
                        coords: np.ndarray = None, edge_types: np.ndarray = None,
                        atom_list: List[str] = None, **kwargs) -> Dict:
        """
        从图数据构建可视化输入
        
        Args:
            node_types: [N] 或 [N, num_types] 节点类型
            edge_index: [2, E] 边索引
            coords: [N, 3] 坐标（可选）
            edge_types: [E] 或 [E, num_bond_types] 边类型（可选）
            atom_list: 原子类型映射表
        """
        atom_list = atom_list or DiffusionModelAdapter.ATOM_TYPES
        
        # 解析节点
        node_types = np.array(node_types)
        if node_types.ndim == 2:
            node_indices = np.argmax(node_types, axis=1)
        else:
            node_indices = node_types.astype(int)
        
        symbols = [atom_list[i] if i < len(atom_list) else 'UNK' for i in node_indices]
        
        # 生成坐标
        n_nodes = len(symbols)
        if coords is None:
            t = np.linspace(0, 4*np.pi, n_nodes)
            coords = np.column_stack([8*np.cos(t), 8*np.sin(t), t])
        
        # 解析边
        edge_index = np.array(edge_index)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T
        
        edges = []
        seen = set()
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            key = (min(u,v), max(u,v))
            if key not in seen:
                seen.add(key)
                bond_type = 0
                if edge_types is not None:
                    et = np.array(edge_types)
                    if et.ndim == 2:
                        bond_type = int(np.argmax(et[i]))
                    elif i < len(et):
                        bond_type = int(et[i])
                edges.append({'src': u, 'dst': v, 'type': bond_type})
        
        # 构建残基列表
        residues = []
        for i, sym in enumerate(symbols):
            residues.append({
                'index': i, 'resseq': i+1, 'chain': 'A', 'resname': sym,
                'site_prob': kwargs.get('site_probs', [0.5]*n_nodes)[i] if i < len(kwargs.get('site_probs', [])) else 0.5,
                'role_name': 'other', 'ca_coord': coords[i]
            })
        
        return {
            'pdb_id': kwargs.get('name', 'Generated'),
            'coords': coords, 'sequence': ''.join(symbols),
            'catalytic_residues': residues, 'edges': edges,
            'node_symbols': symbols, 'triads': [], 'metals': [],
            'metal_centers': [], 'bimetallic_centers': [],
            '_is_molecule': True  # 标记为小分子
        }
    
    @staticmethod
    def from_pyg_data(data, atom_list: List[str] = None) -> Dict:
        """从PyG Data对象构建"""
        import torch
        x = data.x.cpu().numpy() if torch.is_tensor(data.x) else np.array(data.x)
        ei = data.edge_index.cpu().numpy() if torch.is_tensor(data.edge_index) else np.array(data.edge_index)
        ea = data.edge_attr.cpu().numpy() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        
        coords = None
        if hasattr(data, 'pos') and data.pos is not None:
            coords = data.pos.cpu().numpy() if torch.is_tensor(data.pos) else np.array(data.pos)
        
        return DiffusionModelAdapter.from_graph_data(x, ei, coords, ea, atom_list)
    
    @staticmethod
    def from_nanozyme_design(design_path: str) -> Dict:
        """从纳米酶设计输入文件构建"""
        with open(design_path) as f:
            data = json.load(f)
        
        cat_geom = data.get('catalytic_geometry', {})
        
        # 提取三联体
        triads = []
        for t in cat_geom.get('triads', []):
            triads.append({
                'residues': [{'resname': r['name'], 'resseq': i, 'index': i, 'role_name': r.get('role', 'other')}
                            for i, r in enumerate(t.get('residues', []))],
                'distances': t.get('distances', {}),
                'confidence': 0.9
            })
        
        # 提取金属中心
        metals = []
        metal_centers = []
        for i, mc in enumerate(cat_geom.get('metal_centers', [])):
            metals.append({'name': mc['metal_type'], 'coord': [i*5, 0, 0]})
            metal_centers.append({
                'metal': {'name': mc['metal_type']},
                'coordination_number': mc.get('coordination_number', 4),
                'geometry': mc.get('geometry', 'tetrahedral'),
                'ligands': [{'resname': lt, 'distance': 2.0} for lt in mc.get('ligand_types', [])]
            })
        
        return {
            'pdb_id': data.get('source_enzyme', 'Nanozyme'),
            'ec1_prediction': data.get('ec_class', 3),
            'catalytic_residues': [],
            'triads': triads, 'metals': metals,
            'metal_centers': metal_centers,
            'bimetallic_centers': cat_geom.get('bimetallic_centers', [])
        }


# ============================================================================
# 专业软件导出器
# ============================================================================
class ProfessionalExporter:
    """PyMOL, ChimeraX, VMD 脚本导出"""
    
    @staticmethod
    def to_pymol(results: Dict, pdb_path: str, output_path: str,
                 highlight_catalytic: bool = True, show_surface: bool = False):
        """
        导出PyMOL脚本 (.pml)
        
        Usage: pymol -qc output.pml
        """
        pdb_path = Path(pdb_path)
        lines = [
            f"# PyMOL Visualization Script",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"# PDB: {results.get('pdb_id', 'Unknown')}",
            "",
            "# Load structure",
            f"load {pdb_path.name}",
            "",
            "# Basic styling",
            "bg_color white",
            "hide everything",
            "show cartoon",
            "color gray80",
            "set cartoon_fancy_helices, 1",
            "set cartoon_smooth_loops, 1",
            "",
        ]
        
        # 催化残基
        if highlight_catalytic and results.get('catalytic_residues'):
            lines.append("# Catalytic residues")
            for res in results['catalytic_residues'][:20]:
                sel = f"chain {res['chain']} and resi {res['resseq']}"
                role = res.get('role_name', 'other')
                
                color_map = {
                    'nucleophile': 'red', 'general_base': 'cyan',
                    'general_acid': 'yellow', 'electrostatic': 'orange',
                    'metal_ligand': 'blue', 'transition_state_stabilizer': 'purple'
                }
                color = color_map.get(role, 'green')
                
                lines.extend([
                    f"select cat_{res['resseq']}, {sel}",
                    f"show sticks, cat_{res['resseq']}",
                    f"color {color}, cat_{res['resseq']}",
                ])
            lines.append("")
        
        # 金属
        if results.get('metals'):
            lines.append("# Metal centers")
            for m in results['metals']:
                mname = m['name'].upper().rstrip('0123456789')
                lines.extend([
                    f"select metal_{mname}, resn {m['name']}",
                    f"show spheres, metal_{mname}",
                    f"color slate, metal_{mname}",
                    f"set sphere_scale, 0.8, metal_{mname}",
                ])
            lines.append("")
        
        # 三联体距离
        if results.get('triads'):
            lines.append("# Triad distances")
            for i, triad in enumerate(results['triads'][:3]):
                residues = triad.get('residues', [])
                if len(residues) >= 3:
                    for j in range(3):
                        k = (j + 1) % 3
                        r1, r2 = residues[j], residues[k]
                        lines.append(
                            f"distance triad{i}_{j}, "
                            f"chain {r1.get('chain','A')} and resi {r1['resseq']} and name CA, "
                            f"chain {r2.get('chain','A')} and resi {r2['resseq']} and name CA"
                        )
            lines.extend(["color yellow, triad*", "hide labels, triad*", ""])
        
        # 配位键
        if results.get('metal_centers'):
            lines.append("# Coordination bonds")
            for mc in results['metal_centers']:
                metal = mc.get('metal', {})
                for lig in mc.get('ligands', [])[:6]:
                    if 'index' in lig or 'resseq' in lig:
                        resseq = lig.get('resseq', lig.get('index', 0) + 1)
                        lines.append(
                            f"distance coord_{resseq}, resn {metal.get('name','ZN')}, "
                            f"resi {resseq} and name CA"
                        )
            lines.extend(["color slate, coord_*", ""])
        
        # 表面
        if show_surface:
            lines.extend([
                "# Surface",
                "show surface",
                "set transparency, 0.7",
                "color white, surface",
                ""
            ])
        
        # 视图设置
        lines.extend([
            "# Final settings",
            "set ray_shadows, 0",
            "set antialias, 2",
            "set ray_trace_mode, 1",
            "zoom",
            "",
            "# Save image (optional)",
            f"# png {pdb_path.stem}_catalytic.png, width=2400, height=2000, dpi=300, ray=1",
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"✓ PyMOL脚本: {output_path}")
    
    @staticmethod
    def to_chimerax(results: Dict, pdb_path: str, output_path: str,
                    highlight_catalytic: bool = True):
        """
        导出ChimeraX命令脚本 (.cxc)
        
        Usage: chimerax --script output.cxc
        """
        pdb_path = Path(pdb_path)
        lines = [
            f"# ChimeraX Visualization Script",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"# PDB: {results.get('pdb_id', 'Unknown')}",
            "",
            f"open {pdb_path.name}",
            "",
            "# Basic styling",
            "set bgColor white",
            "hide atoms",
            "cartoon",
            "color gray target c",
            "lighting soft",
            "",
        ]
        
        # 催化残基
        if highlight_catalytic and results.get('catalytic_residues'):
            lines.append("# Catalytic residues")
            for res in results['catalytic_residues'][:20]:
                sel = f"/{res['chain']}:{res['resseq']}"
                role = res.get('role_name', 'other')
                
                color_map = {
                    'nucleophile': 'red', 'general_base': 'cyan',
                    'general_acid': 'yellow', 'electrostatic': 'orange',
                    'metal_ligand': 'cornflower blue', 'transition_state_stabilizer': 'purple'
                }
                color = color_map.get(role, 'green')
                
                lines.extend([
                    f"show {sel} atoms",
                    f"style {sel} stick",
                    f"color {sel} {color}",
                ])
            lines.append("")
        
        # 金属
        if results.get('metals'):
            lines.append("# Metal centers")
            for m in results['metals']:
                mname = m['name']
                lines.extend([
                    f"show ::{mname} atoms",
                    f"style ::{mname} sphere",
                    f"color ::{mname} slate blue",
                    f"size ::{mname} atomRadius 1.5",
                ])
            lines.append("")
        
        # 三联体标签
        if results.get('triads'):
            lines.append("# Triad labels")
            for triad in results['triads'][:2]:
                for res in triad.get('residues', [])[:3]:
                    sel = f"/{res.get('chain','A')}:{res['resseq']}"
                    lines.append(f"label {sel} text \"{res['resname']}{res['resseq']}\"")
            lines.extend(["label ontop true", ""])
        
        # 距离测量
        if results.get('triads'):
            lines.append("# Distance measurements")
            triad = results['triads'][0]
            residues = triad.get('residues', [])
            if len(residues) >= 3:
                for i in range(3):
                    j = (i + 1) % 3
                    r1, r2 = residues[i], residues[j]
                    lines.append(
                        f"distance /{r1.get('chain','A')}:{r1['resseq']}@CA "
                        f"/{r2.get('chain','A')}:{r2['resseq']}@CA"
                    )
            lines.append("")
        
        # 最终设置
        lines.extend([
            "# View settings",
            "view",
            "clip off",
            "",
            "# Save (optional)",
            f"# save {pdb_path.stem}_catalytic.png width 2400 height 2000 supersample 3",
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"✓ ChimeraX脚本: {output_path}")
    
    @staticmethod
    def to_vmd(results: Dict, pdb_path: str, output_path: str):
        """
        导出VMD Tcl脚本 (.tcl)
        
        Usage: vmd -e output.tcl
        """
        pdb_path = Path(pdb_path)
        lines = [
            f"# VMD Visualization Script",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"mol new {pdb_path.name} type pdb",
            "",
            "# Representation",
            "mol delrep 0 top",
            "mol representation NewCartoon",
            "mol color ColorID 2",
            "mol addrep top",
            "",
        ]
        
        # 催化残基
        if results.get('catalytic_residues'):
            lines.append("# Catalytic residues")
            resids = ' '.join([str(r['resseq']) for r in results['catalytic_residues'][:20]])
            lines.extend([
                f"mol representation Licorice",
                f"mol selection \"resid {resids}\"",
                f"mol color Name",
                f"mol addrep top",
                "",
            ])
        
        # 金属
        if results.get('metals'):
            names = ' '.join([m['name'] for m in results['metals']])
            lines.extend([
                "# Metals",
                f"mol representation VDW 1.0",
                f"mol selection \"resname {names}\"",
                f"mol color ColorID 0",
                f"mol addrep top",
                "",
            ])
        
        lines.extend([
            "# Display settings",
            "display projection Orthographic",
            "display depthcue off",
            "color Display Background white",
            "axes location Off",
            "",
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"✓ VMD脚本: {output_path}")
    
    @staticmethod
    def to_pdb(results: Dict, coords: np.ndarray, output_path: str,
               node_symbols: List[str] = None):
        """导出简化PDB文件（用于小分子/生成结构）"""
        lines = [
            f"REMARK   Generated by NanozymeVisualizer",
            f"REMARK   {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        
        symbols = node_symbols or results.get('node_symbols', ['C'] * len(coords))
        
        for i, (coord, sym) in enumerate(zip(coords, symbols)):
            atom_name = sym[:4].ljust(4)
            lines.append(
                f"ATOM  {i+1:5d} {atom_name} MOL A{i+1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           {sym[:2].rjust(2)}"
            )
        
        # 边/键
        if results.get('edges'):
            lines.append("CONECT")
            for edge in results['edges']:
                lines.append(f"CONECT{edge['src']+1:5d}{edge['dst']+1:5d}")
        
        lines.append("END")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"✓ PDB文件: {output_path}")


# ============================================================================
# 2D可视化器
# ============================================================================
class Visualizer2D:
    """2D可视化"""
    
    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
    
    def plot_molecular_graph(self, results: Dict, output_path: str = None) -> plt.Figure:
        """分子图风格可视化"""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(f"Molecular Graph - {results.get('pdb_id', 'Unknown')}",
                    fontsize=14, fontweight='bold', style='italic')
        
        is_molecule = results.get('_is_molecule', False)
        catalytic = results.get('catalytic_residues', [])
        metals = results.get('metals', [])
        edges = results.get('edges', [])
        triads = results.get('triads', [])
        
        if not catalytic and not metals:
            ax.text(0.5, 0.5, "No data to display", ha='center', va='center')
            ax.axis('off')
            return fig
        
        G = nx.Graph()
        node_data = {}
        
        # 添加节点
        for res in catalytic[:30]:
            nid = f"{res['resname']}{res['resseq']}"
            G.add_node(nid)
            node_data[nid] = {
                'type': 'residue', 'resname': res['resname'],
                'prob': res.get('site_prob', 0.5), 'role': res.get('role_name', 'other'),
                'index': res.get('index', res['resseq']-1)
            }
        
        for i, m in enumerate(metals):
            nid = f"{m['name']}_{i+1}"
            G.add_node(nid)
            node_data[nid] = {'type': 'metal', 'name': m['name']}
        
        # 添加边
        if is_molecule and edges:
            # 小分子：使用edges
            for edge in edges:
                s, d = edge['src'], edge['dst']
                n1 = f"{catalytic[s]['resname']}{catalytic[s]['resseq']}" if s < len(catalytic) else None
                n2 = f"{catalytic[d]['resname']}{catalytic[d]['resseq']}" if d < len(catalytic) else None
                if n1 and n2 and n1 in G.nodes() and n2 in G.nodes():
                    G.add_edge(n1, n2, type=edge.get('type', 0))
        else:
            # 蛋白质：使用三联体
            for triad in triads[:5]:
                residues = triad.get('residues', [])
                for j in range(len(residues)):
                    for k in range(j+1, len(residues)):
                        n1 = f"{residues[j]['resname']}{residues[j]['resseq']}"
                        n2 = f"{residues[k]['resname']}{residues[k]['resseq']}"
                        if n1 in G.nodes() and n2 in G.nodes():
                            G.add_edge(n1, n2, type='triad')
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, "No graph to display", ha='center', va='center')
            ax.axis('off')
            return fig
        
        # 布局
        pos = nx.kamada_kawai_layout(G) if len(G.nodes()) < 20 else nx.spring_layout(G, k=2, seed=42)
        
        # 绘制边
        bond_symbols = {0: '-', 1: '=', 2: '≡', 3: '*', 'triad': '-', 'coordination': '→'}
        for u, v, data in G.edges(data=True):
            x, y = [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]]
            etype = data.get('type', 0)
            
            style = '--' if etype == 'coordination' else '-'
            color = '#5C7AEA' if etype == 'coordination' else '#666666'
            ax.plot(x, y, color=color, linewidth=2, linestyle=style, zorder=1)
            
            # 边标签
            mx, my = (x[0]+x[1])/2, (y[0]+y[1])/2
            symbol = bond_symbols.get(etype, '-')
            ax.annotate(symbol, (mx, my), fontsize=9, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#CCC'),
                       color='#CC6666', fontweight='bold', zorder=2)
        
        # 绘制节点
        for node in G.nodes():
            data = node_data.get(node, {})
            x, y = pos[node]
            
            if data.get('type') == 'metal':
                name = data.get('name', 'M').upper().rstrip('0123456789')
                color = METAL_COLORS.get(name, METAL_COLORS['DEFAULT'])
                circle = plt.Circle((x, y), 0.08, color=color, ec='#333', lw=2, zorder=3)
                ax.add_patch(circle)
                ax.annotate(name, (x, y), fontsize=11, ha='center', va='center',
                           fontweight='bold', color='white', zorder=4)
            else:
                resname = data.get('resname', 'UNK')
                role = data.get('role', 'other')
                color = ROLE_COLORS.get(role, AA_COLORS.get(resname, '#CCC')) if role != 'other' else AA_COLORS.get(resname, '#CCC')
                
                circle = plt.Circle((x, y), 0.06, color=color, ec='#666', lw=1.5, zorder=3)
                ax.add_patch(circle)
                ax.annotate(resname[:3], (x, y), fontsize=9, ha='center', va='center',
                           fontweight='bold', zorder=4)
        
        # 图例
        if not is_molecule:
            legend = [
                mpatches.Patch(color='#FF6B6B', label='Nucleophile'),
                mpatches.Patch(color='#4ECDC4', label='General Base'),
                mpatches.Patch(color='#FFE66D', label='Electrostatic'),
                mpatches.Patch(color='#5C7AEA', label='Metal'),
            ]
            ax.legend(handles=legend, loc='upper right', fontsize=9)
        
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ 2D图: {output_path}")
        
        return fig
    
    def plot_triads(self, results: Dict, output_path: str = None) -> plt.Figure:
        """三联体几何"""
        triads = results.get('triads', [])
        n = min(len(triads), 4) or 1
        
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        axes = [axes] if n == 1 else axes
        
        if not triads:
            axes[0].text(0.5, 0.5, "No triads", ha='center', va='center')
            axes[0].axis('off')
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            return fig
        
        for idx, (triad, ax) in enumerate(zip(triads[:n], axes)):
            residues = triad.get('residues', [])
            distances = triad.get('distances', {})
            conf = triad.get('confidence', 0)
            
            ax.set_title(f"Triad {idx+1} (conf: {conf:.2f})", fontsize=11, fontweight='bold')
            
            if len(residues) < 3:
                ax.text(0.5, 0.5, "Incomplete", ha='center', va='center')
                ax.axis('off')
                continue
            
            angles = [90, 210, 330]
            positions = [(np.cos(np.radians(a)), np.sin(np.radians(a))) for a in angles]
            
            # 边
            for i in range(3):
                j = (i + 1) % 3
                ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]],
                       color='#666', linewidth=2, zorder=1)
                
                mx = (positions[i][0] + positions[j][0]) / 2
                my = (positions[i][1] + positions[j][1]) / 2
                key = f"{residues[i]['resname']}-{residues[j]['resname']}"
                dist = distances.get(key, distances.get(f"{residues[j]['resname']}-{residues[i]['resname']}", 0))
                if dist:
                    ax.annotate(f"{dist:.1f}Å", (mx, my), fontsize=9, ha='center',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CCC'))
            
            # 节点
            for i, res in enumerate(residues[:3]):
                role = res.get('role_name', 'other')
                color = ROLE_COLORS.get(role, AA_COLORS.get(res['resname'], '#CCC'))
                circle = plt.Circle(positions[i], 0.15, color=color, ec='#333', lw=2, zorder=3)
                ax.add_patch(circle)
                ax.annotate(f"{res['resname']}\n{res['resseq']}", positions[i],
                           fontsize=9, ha='center', va='center', fontweight='bold', zorder=4)
            
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            ax.set_aspect('equal')
            ax.axis('off')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ 三联体图: {output_path}")
        
        return fig
    
    def plot_metal_centers(self, results: Dict, output_path: str = None) -> plt.Figure:
        """金属中心"""
        mcs = results.get('metal_centers', [])
        bmcs = results.get('bimetallic_centers', [])
        n = len(mcs) + len(bmcs) or 1
        
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        
        if not mcs and not bmcs:
            axes[0].text(0.5, 0.5, "No metal centers", ha='center', va='center')
            axes[0].axis('off')
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            return fig
        
        idx = 0
        # 单金属
        for mc in mcs:
            if idx >= len(axes):
                break
            ax = axes[idx]
            metal = mc.get('metal', {})
            ligands = mc.get('ligands', [])
            geom = mc.get('geometry', 'tetrahedral')
            cn = mc.get('coordination_number', 4)
            mname = metal.get('name', 'M').upper().rstrip('0123456789')
            
            ax.set_title(f"{mname} - {geom} (CN={cn})", fontsize=11, fontweight='bold')
            
            # 中心
            color = METAL_COLORS.get(mname, METAL_COLORS['DEFAULT'])
            ax.scatter([0], [0], s=800, c=[color], edgecolors='#333', linewidth=2, zorder=5)
            ax.annotate(mname, (0, 0), fontsize=12, ha='center', va='center',
                       fontweight='bold', color='white', zorder=6)
            
            # 配体
            n_lig = len(ligands)
            for i, lig in enumerate(ligands[:6]):
                angle = 2 * np.pi * i / max(n_lig, 1)
                lx, ly = 1.5 * np.cos(angle), 1.5 * np.sin(angle)
                
                ax.plot([0, lx], [0, ly], color='#5C7AEA', linewidth=2, linestyle='--', zorder=1)
                lcolor = AA_COLORS.get(lig.get('resname', 'UNK'), '#CCC')
                ax.scatter([lx], [ly], s=400, c=[lcolor], edgecolors='#666', linewidth=1.5, zorder=3)
                ax.annotate(lig.get('resname', '?'), (lx, ly), fontsize=9,
                           ha='center', va='center', fontweight='bold', zorder=4)
                
                dist = lig.get('distance', 0)
                if dist:
                    ax.annotate(f"{dist:.1f}Å", (lx/2, ly/2), fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
            
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal')
            ax.axis('off')
            idx += 1
        
        # 双金属
        for bmc in bmcs:
            if idx >= len(axes):
                break
            ax = axes[idx]
            m1, m2 = bmc.get('metal1', {}), bmc.get('metal2', {})
            dist = bmc.get('distance', 0)
            pattern = bmc.get('pattern', 'unknown')
            bridging = bmc.get('bridging_residues', [])
            
            m1n = m1.get('name', 'M1').upper().rstrip('0123456789')
            m2n = m2.get('name', 'M2').upper().rstrip('0123456789')
            
            ax.set_title(f"Bimetallic: {m1n}-{m2n} ({pattern})", fontsize=10, fontweight='bold')
            
            ax.scatter([-0.8], [0], s=800, c=[METAL_COLORS.get(m1n, '#6C757D')],
                      edgecolors='#333', linewidth=2, zorder=5)
            ax.scatter([0.8], [0], s=800, c=[METAL_COLORS.get(m2n, '#6C757D')],
                      edgecolors='#333', linewidth=2, zorder=5)
            ax.annotate(m1n, (-0.8, 0), ha='center', va='center', fontsize=11,
                       fontweight='bold', color='white', zorder=6)
            ax.annotate(m2n, (0.8, 0), ha='center', va='center', fontsize=11,
                       fontweight='bold', color='white', zorder=6)
            
            ax.plot([-0.8, 0.8], [0, 0], color='#333', linewidth=3, zorder=1)
            ax.annotate(f"{dist:.1f}Å", (0, 0.15), ha='center', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='yellow', alpha=0.8))
            
            for i, br in enumerate(bridging[:3]):
                y_off = 0.8 if i % 2 == 0 else -0.8
                bcolor = AA_COLORS.get(br.get('resname', 'UNK'), '#CCC')
                ax.scatter([0], [y_off], s=300, c=[bcolor], edgecolors='#666', zorder=3)
                ax.annotate(br.get('resname', '?'), (0, y_off), fontsize=8,
                           ha='center', va='center', fontweight='bold', zorder=4)
                ax.plot([-0.8, 0, 0.8], [0, y_off, 0], color='#888', linewidth=1.5, linestyle=':', zorder=1)
            
            ax.set_xlim(-2, 2)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.axis('off')
            idx += 1
        
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ 金属中心图: {output_path}")
        
        return fig


# ============================================================================
# 3D可视化器
# ============================================================================
class Visualizer3D:
    """3D可视化"""
    
    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
    
    def plot_active_site(self, results: Dict, coords: np.ndarray = None,
                         output_path: str = None) -> plt.Figure:
        """3D活性位点"""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"3D Active Site - {results.get('pdb_id', 'Unknown')}",
                    fontsize=14, fontweight='bold')
        
        catalytic = results.get('catalytic_residues', [])
        metals = results.get('metals', [])
        
        if coords is None:
            coords = self._gen_coords(catalytic, metals)
        
        # 催化残基
        for res in catalytic[:20]:
            idx = res.get('index', 0)
            if idx < len(coords):
                role = res.get('role_name', 'other')
                color = ROLE_COLORS.get(role, AA_COLORS.get(res['resname'], '#CCC'))
                ax.scatter([coords[idx, 0]], [coords[idx, 1]], [coords[idx, 2]],
                          c=[color], s=300*res.get('site_prob', 0.5), alpha=0.8,
                          edgecolors='#333', linewidth=1.5)
                ax.text(coords[idx, 0], coords[idx, 1], coords[idx, 2] + 1,
                       f"{res['resname']}{res['resseq']}", fontsize=8, ha='center')
        
        # 金属
        for i, m in enumerate(metals):
            mc = np.array(m.get('coord', [0, 0, 0]))
            mname = m.get('name', 'M').upper().rstrip('0123456789')
            color = METAL_COLORS.get(mname, METAL_COLORS['DEFAULT'])
            ax.scatter([mc[0]], [mc[1]], [mc[2]], c=[color], s=500, marker='D',
                      edgecolors='#000', linewidth=2, alpha=0.9)
            ax.text(mc[0], mc[1], mc[2] + 2, mname, fontsize=10, fontweight='bold', ha='center')
        
        # 三联体连线
        for triad in results.get('triads', [])[:3]:
            self._draw_triad_3d(ax, triad, coords)
        
        # 配位键
        for mc in results.get('metal_centers', []):
            self._draw_coord_bonds_3d(ax, mc, coords, metals)
        
        self._style_axes(ax)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ 3D活性位点: {output_path}")
        
        return fig
    
    def plot_metal_polyhedra(self, results: Dict, coords: np.ndarray = None,
                             output_path: str = None) -> plt.Figure:
        """3D金属配位多面体"""
        mcs = results.get('metal_centers', [])
        bmcs = results.get('bimetallic_centers', [])
        n = len(mcs) + len(bmcs) or 1
        
        cols = min(2, n)
        rows = (n + 1) // 2
        fig = plt.figure(figsize=(6*cols, 6*rows))
        
        idx = 0
        
        # 单金属
        for mc in mcs:
            idx += 1
            if idx > rows * cols:
                break
            ax = fig.add_subplot(rows, cols, idx, projection='3d')
            
            metal = mc.get('metal', {})
            ligands = mc.get('ligands', [])
            geom = mc.get('geometry', 'tetrahedral')
            cn = mc.get('coordination_number', 4)
            mname = metal.get('name', 'M').upper().rstrip('0123456789')
            
            ax.set_title(f"{mname} - {geom} (CN={cn})", fontsize=11, fontweight='bold')
            
            # 金属中心
            color = METAL_COLORS.get(mname, METAL_COLORS['DEFAULT'])
            ax.scatter([0], [0], [0], c=[color], s=600, edgecolors='#000', linewidth=2, zorder=10)
            ax.text(0, 0, 0.3, mname, fontsize=12, ha='center', fontweight='bold', zorder=11)
            
            # 几何模板
            gd = COORD_GEOMETRIES.get(geom, COORD_GEOMETRIES['tetrahedral'])
            verts = gd['v'] * 2.5
            faces = gd['f']
            
            n_lig = min(len(ligands), len(verts))
            
            # 配体
            for i in range(n_lig):
                lig = ligands[i]
                v = verts[i]
                lcolor = AA_COLORS.get(lig.get('resname', 'UNK'), '#CCC')
                ax.scatter([v[0]], [v[1]], [v[2]], c=[lcolor], s=300, edgecolors='#666', linewidth=1.5)
                ax.text(v[0]*1.2, v[1]*1.2, v[2]*1.2, lig.get('resname', '?'),
                       fontsize=9, ha='center', fontweight='bold')
                ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color='#5C7AEA', linewidth=2, linestyle='--')
                
                dist = lig.get('distance', 0)
                if dist:
                    ax.text(v[0]/2, v[1]/2, v[2]/2, f"{dist:.1f}Å", fontsize=7)
            
            # 多面体面
            if n_lig >= 3:
                for face in faces:
                    if all(f < n_lig for f in face):
                        fv = verts[face]
                        poly = Poly3DCollection([fv], alpha=0.15, facecolor=color,
                                               edgecolor='#333', linewidth=0.5)
                        ax.add_collection3d(poly)
            
            self._style_axes(ax, lim=4)
        
        # 双金属
        for bmc in bmcs:
            idx += 1
            if idx > rows * cols:
                break
            ax = fig.add_subplot(rows, cols, idx, projection='3d')
            
            m1, m2 = bmc.get('metal1', {}), bmc.get('metal2', {})
            dist = bmc.get('distance', 3.5)
            pattern = bmc.get('pattern', 'unknown')
            bridging = bmc.get('bridging_residues', [])
            
            m1n = m1.get('name', 'M1').upper().rstrip('0123456789')
            m2n = m2.get('name', 'M2').upper().rstrip('0123456789')
            
            ax.set_title(f"Bimetallic: {m1n}-{m2n}\n({pattern}, {dist:.1f}Å)", fontsize=10, fontweight='bold')
            
            m1p, m2p = np.array([-dist/2, 0, 0]), np.array([dist/2, 0, 0])
            
            ax.scatter([m1p[0]], [m1p[1]], [m1p[2]], c=[METAL_COLORS.get(m1n, '#6C757D')],
                      s=500, edgecolors='#000', linewidth=2, zorder=10)
            ax.scatter([m2p[0]], [m2p[1]], [m2p[2]], c=[METAL_COLORS.get(m2n, '#6C757D')],
                      s=500, edgecolors='#000', linewidth=2, zorder=10)
            ax.text(m1p[0], m1p[1], m1p[2]+0.5, m1n, fontsize=11, ha='center', fontweight='bold', zorder=11)
            ax.text(m2p[0], m2p[1], m2p[2]+0.5, m2n, fontsize=11, ha='center', fontweight='bold', zorder=11)
            
            ax.plot([m1p[0], m2p[0]], [m1p[1], m2p[1]], [m1p[2], m2p[2]], color='#333', linewidth=3, zorder=5)
            ax.text(0, 0, -0.5, f"{dist:.1f}Å", fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            for i, br in enumerate(bridging[:4]):
                angle = np.pi/2 + i * np.pi/2
                bp = np.array([0, 2*np.cos(angle), 2*np.sin(angle)])
                bcolor = AA_COLORS.get(br.get('resname', 'UNK'), '#CCC')
                ax.scatter([bp[0]], [bp[1]], [bp[2]], c=[bcolor], s=250, edgecolors='#666', linewidth=1.5)
                ax.text(bp[0], bp[1]*1.3, bp[2]*1.3, br.get('resname', '?'), fontsize=8, ha='center')
                ax.plot([m1p[0], bp[0]], [m1p[1], bp[1]], [m1p[2], bp[2]], color='#888', linewidth=1.5, linestyle=':')
                ax.plot([m2p[0], bp[0]], [m2p[1], bp[1]], [m2p[2], bp[2]], color='#888', linewidth=1.5, linestyle=':')
            
            self._style_axes(ax, lim=4)
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ 3D配位多面体: {output_path}")
        
        return fig
    
    def plot_interactive(self, results: Dict, coords: np.ndarray = None,
                         output_path: str = None):
        """Plotly交互式3D"""
        if not PLOTLY_AVAILABLE:
            print("需要plotly: pip install plotly")
            return None
        
        catalytic = results.get('catalytic_residues', [])
        metals = results.get('metals', [])
        triads = results.get('triads', [])
        
        if coords is None:
            coords = self._gen_coords(catalytic, metals)
        
        fig = go.Figure()
        
        # 催化残基
        cx, cy, cz, colors, texts, sizes = [], [], [], [], [], []
        for res in catalytic[:30]:
            idx = res.get('index', 0)
            if idx < len(coords):
                cx.append(coords[idx, 0])
                cy.append(coords[idx, 1])
                cz.append(coords[idx, 2])
                role = res.get('role_name', 'other')
                colors.append(ROLE_COLORS.get(role, AA_COLORS.get(res['resname'], '#CCC')))
                texts.append(f"{res['resname']}{res['resseq']}<br>Role: {role}<br>Prob: {res.get('site_prob',0):.2f}")
                sizes.append(15 + 20 * res.get('site_prob', 0.5))
        
        if cx:
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz, mode='markers+text',
                marker=dict(size=sizes, color=colors, line=dict(width=2, color='#333')),
                text=[f"{catalytic[i]['resname']}{catalytic[i]['resseq']}" for i in range(len(cx))],
                textposition='top center', hovertext=texts, hoverinfo='text',
                name='Catalytic Residues'
            ))
        
        # 金属
        for i, m in enumerate(metals):
            mc = np.array(m.get('coord', [0, 0, 0]))
            mname = m.get('name', 'M').upper().rstrip('0123456789')
            color = METAL_COLORS.get(mname, '#6C757D')
            fig.add_trace(go.Scatter3d(
                x=[mc[0]], y=[mc[1]], z=[mc[2]], mode='markers+text',
                marker=dict(size=25, color=color, symbol='diamond', line=dict(width=3, color='#000')),
                text=[mname], textposition='top center', name=f'Metal: {mname}'
            ))
        
        # 三联体连线
        for triad in triads[:3]:
            residues = triad.get('residues', [])
            if len(residues) >= 3:
                tc = []
                for res in residues[:3]:
                    idx = res.get('index', 0)
                    if idx < len(coords):
                        tc.append(coords[idx])
                if len(tc) == 3:
                    tc = np.array(tc)
                    for i in range(3):
                        j = (i + 1) % 3
                        fig.add_trace(go.Scatter3d(
                            x=[tc[i,0], tc[j,0]], y=[tc[i,1], tc[j,1]], z=[tc[i,2], tc[j,2]],
                            mode='lines', line=dict(color='#666', width=4), showlegend=False
                        ))
        
        fig.update_layout(
            title=dict(text=f"3D Interactive - {results.get('pdb_id', 'Unknown')}", font=dict(size=18)),
            scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)', aspectmode='data'),
            showlegend=True, legend=dict(x=0.02, y=0.98), margin=dict(l=0, r=0, t=50, b=0)
        )
        
        if output_path:
            if output_path.endswith('.html'):
                fig.write_html(output_path)
            else:
                fig.write_image(output_path)
            print(f"✓ 交互式3D: {output_path}")
        
        return fig
    
    def export_rotation_gif(self, results: Dict, coords: np.ndarray = None,
                           output_path: str = "rotation.gif", n_frames: int = 36):
        """导出旋转动画GIF"""
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("需要Pillow: pip install Pillow")
            return
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        catalytic = results.get('catalytic_residues', [])[:10]
        metals = results.get('metals', [])
        if coords is None:
            coords = self._gen_coords(catalytic, metals)
        
        for res in catalytic:
            idx = res.get('index', 0)
            if idx < len(coords):
                color = ROLE_COLORS.get(res.get('role_name', 'other'), '#CCC')
                ax.scatter([coords[idx, 0]], [coords[idx, 1]], [coords[idx, 2]],
                          c=[color], s=200*res.get('site_prob', 0.5), alpha=0.8, edgecolors='#333')
        
        for m in metals:
            mc = np.array(m.get('coord', [0, 0, 0]))
            mname = m.get('name', 'M').upper().rstrip('0123456789')
            ax.scatter([mc[0]], [mc[1]], [mc[2]], c=[METAL_COLORS.get(mname, '#6C757D')],
                      s=400, marker='D', edgecolors='#000', linewidth=2)
        
        ax.set_title(f"3D Rotation - {results.get('pdb_id', 'Unknown')}", fontsize=14)
        self._style_axes(ax)
        
        def update(frame):
            ax.view_init(elev=20, azim=frame * 360 / n_frames)
            return []
        
        anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
        anim.save(output_path, writer=PillowWriter(fps=10))
        print(f"✓ 旋转动画: {output_path}")
        plt.close()
    
    def _gen_coords(self, catalytic, metals):
        """生成模拟坐标"""
        n = max(max([r.get('index', 0) for r in catalytic] + [0]) + 1, 20)
        t = np.linspace(0, 4*np.pi, n)
        return np.column_stack([10*np.cos(t), 10*np.sin(t), t*2])
    
    def _draw_triad_3d(self, ax, triad, coords):
        """绘制三联体连接"""
        residues = triad.get('residues', [])
        for i in range(len(residues)):
            for j in range(i+1, len(residues)):
                ii, jj = residues[i].get('index', 0), residues[j].get('index', 0)
                if ii < len(coords) and jj < len(coords):
                    ax.plot([coords[ii,0], coords[jj,0]], [coords[ii,1], coords[jj,1]],
                           [coords[ii,2], coords[jj,2]], color='#FF6B6B', linewidth=2, alpha=0.7)
    
    def _draw_coord_bonds_3d(self, ax, mc, coords, metals):
        """绘制配位键"""
        metal = mc.get('metal', {})
        ligands = mc.get('ligands', [])
        mcoord = None
        for m in metals:
            if m.get('name') == metal.get('name'):
                mcoord = np.array(m.get('coord', [0, 0, 0]))
                break
        if mcoord is None:
            return
        for lig in ligands[:6]:
            idx = lig.get('index', 0)
            if idx < len(coords):
                ax.plot([mcoord[0], coords[idx,0]], [mcoord[1], coords[idx,1]],
                       [mcoord[2], coords[idx,2]], color='#5C7AEA', linewidth=1.5, linestyle='--', alpha=0.6)
    
    def _style_axes(self, ax, lim=None):
        """美化3D坐标轴"""
        ax.set_xlabel('X (Å)', fontsize=9)
        ax.set_ylabel('Y (Å)', fontsize=9)
        ax.set_zlabel('Z (Å)', fontsize=9)
        if lim:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)


# ============================================================================
# 统一接口
# ============================================================================
class NanozymeVisualizer:
    """
    纳米酶可视化统一接口
    
    用法:
        viz = NanozymeVisualizer()
        
        # 从预测结果可视化
        viz.visualize(results, coords, output_dir="./output")
        
        # 从扩散模型输出可视化
        viz.visualize_diffusion(node_types, edge_index, coords, output_dir="./output")
        
        # 导出专业软件脚本
        viz.export_professional(results, "protein.pdb", output_dir="./output")
    """
    
    def __init__(self, figsize_2d=(12, 10), figsize_3d=(12, 10)):
        self.viz2d = Visualizer2D(figsize_2d)
        self.viz3d = Visualizer3D(figsize_3d)
        self.adapter = DiffusionModelAdapter()
        self.exporter = ProfessionalExporter()
    
    def visualize(self, results: Dict, coords: np.ndarray = None,
                  output_dir: str = "./output", prefix: str = "",
                  modes: List[str] = None):
        """
        完整可视化
        
        Args:
            results: predict()输出 或 适配后的字典
            coords: CA坐标 [N, 3]
            output_dir: 输出目录
            prefix: 文件名前缀
            modes: 可视化模式列表，默认全部
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = prefix or results.get('pdb_id', 'output')
        modes = modes or ['2d_graph', '2d_triad', '2d_metal', '3d_site', '3d_metal', 'interactive']
        
        print(f"\n{'='*50}")
        print(f"可视化: {prefix}")
        print(f"{'='*50}")
        
        if '2d_graph' in modes:
            self.viz2d.plot_molecular_graph(results, output_dir / f"{prefix}_2d_graph.png")
        
        if '2d_triad' in modes:
            self.viz2d.plot_triads(results, output_dir / f"{prefix}_2d_triads.png")
        
        if '2d_metal' in modes:
            self.viz2d.plot_metal_centers(results, output_dir / f"{prefix}_2d_metals.png")
        
        if '3d_site' in modes:
            self.viz3d.plot_active_site(results, coords, output_dir / f"{prefix}_3d_site.png")
        
        if '3d_metal' in modes:
            self.viz3d.plot_metal_polyhedra(results, coords, output_dir / f"{prefix}_3d_polyhedra.png")
        
        if 'interactive' in modes and PLOTLY_AVAILABLE:
            self.viz3d.plot_interactive(results, coords, output_dir / f"{prefix}_interactive.html")
        
        if 'rotation' in modes:
            self.viz3d.export_rotation_gif(results, coords, output_dir / f"{prefix}_rotation.gif")
        
        print(f"\n✓ 所有可视化已保存到: {output_dir}")
    
    def visualize_diffusion(self, node_types: np.ndarray, edge_index: np.ndarray,
                           coords: np.ndarray = None, edge_types: np.ndarray = None,
                           atom_list: List[str] = None, output_dir: str = "./output",
                           name: str = "generated", **kwargs):
        """
        可视化扩散模型生成的结构
        
        Args:
            node_types: 节点类型 [N] 或 [N, num_types]
            edge_index: 边索引 [2, E]
            coords: 坐标 [N, 3]
            edge_types: 边类型 [E] 或 [E, num_bond_types]
            atom_list: 原子类型映射
            output_dir: 输出目录
            name: 结构名称
        """
        results = self.adapter.from_graph_data(
            node_types, edge_index, coords, edge_types, atom_list, name=name, **kwargs
        )
        
        self.visualize(results, results.get('coords'), output_dir, name,
                      modes=['2d_graph', '3d_site', 'interactive'])
        
        return results
    
    def visualize_pyg(self, data, atom_list: List[str] = None,
                      output_dir: str = "./output", name: str = "pyg_mol"):
        """可视化PyG Data对象"""
        results = self.adapter.from_pyg_data(data, atom_list)
        self.visualize(results, results.get('coords'), output_dir, name,
                      modes=['2d_graph', '3d_site'])
        return results
    
    def visualize_rfdiffusion(self, output_path: str, output_dir: str = "./output"):
        """可视化RFdiffusion输出"""
        results = self.adapter.from_rfdiffusion(output_path)
        self.visualize(results, results.get('coords'), output_dir, results.get('pdb_id'))
        return results
    
    def visualize_nanozyme_design(self, design_path: str, output_dir: str = "./output"):
        """可视化纳米酶设计输入"""
        results = self.adapter.from_nanozyme_design(design_path)
        self.visualize(results, None, output_dir, results.get('pdb_id'),
                      modes=['2d_triad', '2d_metal', '3d_metal'])
        return results
    
    def export_professional(self, results: Dict, pdb_path: str,
                           output_dir: str = "./output", prefix: str = None):
        """导出专业软件脚本"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = prefix or results.get('pdb_id', 'output')
        
        print(f"\n导出专业软件脚本...")
        self.exporter.to_pymol(results, pdb_path, output_dir / f"{prefix}.pml")
        self.exporter.to_chimerax(results, pdb_path, output_dir / f"{prefix}.cxc")
        self.exporter.to_vmd(results, pdb_path, output_dir / f"{prefix}.tcl")
        
        # 如果是生成的小分子，导出简化PDB
        if results.get('_is_molecule') and results.get('coords') is not None:
            self.exporter.to_pdb(results, results['coords'], output_dir / f"{prefix}_generated.pdb",
                                results.get('node_symbols'))


# ============================================================================
# 使用示例
# ============================================================================
def demo():
    """完整演示"""
    print("="*60)
    print("纳米酶可视化模块 v2.0 - 完整演示")
    print("="*60)
    
    # 模拟催化三联体预测结果
    mock_results = {
        'pdb_id': '1TRY',
        'num_residues': 223,
        'ec1_prediction': 3,
        'ec1_confidence': 0.92,
        'catalytic_residues': [
            {'index': 57, 'chain': 'A', 'resseq': 57, 'resname': 'SER', 'site_prob': 0.95, 'role_name': 'nucleophile'},
            {'index': 102, 'chain': 'A', 'resseq': 102, 'resname': 'HIS', 'site_prob': 0.91, 'role_name': 'general_base'},
            {'index': 195, 'chain': 'A', 'resseq': 195, 'resname': 'ASP', 'site_prob': 0.88, 'role_name': 'electrostatic'},
            {'index': 189, 'chain': 'A', 'resseq': 189, 'resname': 'GLY', 'site_prob': 0.72, 'role_name': 'transition_state_stabilizer'},
        ],
        'triads': [{
            'residues': [
                {'resname': 'SER', 'resseq': 57, 'index': 57, 'chain': 'A', 'role_name': 'nucleophile'},
                {'resname': 'HIS', 'resseq': 102, 'index': 102, 'chain': 'A', 'role_name': 'general_base'},
                {'resname': 'ASP', 'resseq': 195, 'index': 195, 'chain': 'A', 'role_name': 'electrostatic'},
            ],
            'distances': {'SER-HIS': 3.2, 'HIS-ASP': 2.8, 'SER-ASP': 7.5},
            'confidence': 0.91
        }],
        'metals': [
            {'name': 'ZN', 'coord': [15, 10, 8]},
        ],
        'metal_centers': [{
            'metal': {'name': 'ZN'},
            'coordination_number': 4,
            'geometry': 'tetrahedral',
            'ligands': [
                {'resname': 'HIS', 'index': 94, 'distance': 2.1},
                {'resname': 'HIS', 'index': 96, 'distance': 2.0},
                {'resname': 'CYS', 'index': 99, 'distance': 2.3},
                {'resname': 'CYS', 'index': 102, 'distance': 2.2},
            ]
        }],
        'bimetallic_centers': [{
            'metal1': {'name': 'ZN'},
            'metal2': {'name': 'MG'},
            'distance': 3.8,
            'pattern': 'phosphodiesterase',
            'bridging_residues': [
                {'resname': 'ASP', 'resseq': 120},
                {'resname': 'GLU', 'resseq': 152},
            ]
        }],
    }
    
    # 模拟坐标
    np.random.seed(42)
    mock_coords = np.random.randn(200, 3) * 15
    mock_coords[57] = [0, 0, 0]
    mock_coords[102] = [3, 1, 0.5]
    mock_coords[195] = [5, 3, 1]
    
    # 创建可视化器
    viz = NanozymeVisualizer()
    
    # 1. 完整可视化
    print("\n[1] 催化位点可视化...")
    viz.visualize(mock_results, mock_coords, "./demo_output", "catalytic_site")
    
    # 2. 扩散模型输出可视化
    print("\n[2] 扩散模型输出可视化...")
    node_types = np.array([0, 0, 1, 0, 0, 2, 0, 1])  # 0=C, 1=N, 2=O
    edge_index = np.array([[0,1,1,2,2,3,3,4,4,5,5,6,6,7],
                           [1,0,2,1,3,2,4,3,5,4,6,5,7,6]])
    edge_types = np.array([0,0,1,1,0,0,0,0,0,0,1,1,0,0])  # 0=单键, 1=双键
    mol_coords = np.random.randn(8, 3) * 3
    
    viz.visualize_diffusion(node_types, edge_index, mol_coords, edge_types,
                           atom_list=['C', 'N', 'O'], output_dir="./demo_output",
                           name="diffusion_mol")
    
    # 3. 专业软件导出
    print("\n[3] 导出PyMOL/ChimeraX脚本...")
    viz.export_professional(mock_results, "1TRY.pdb", "./demo_output", "1TRY")
    
    print("\n" + "="*60)
    print("✓ 演示完成！所有文件保存在 ./demo_output/")
    print("="*60)


if __name__ == "__main__":
    demo()