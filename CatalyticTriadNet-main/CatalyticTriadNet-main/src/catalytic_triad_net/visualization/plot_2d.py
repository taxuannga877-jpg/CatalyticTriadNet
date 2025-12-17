#!/usr/bin/env python3
"""
2D可视化模块
包含: 分子图、三联体图、金属中心图、完整报告
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 颜色配置
AA_COLORS = {
    'SER': '#FF6B6B', 'CYS': '#FF8E8E', 'THR': '#FFA5A5',
    'HIS': '#4ECDC4', 'LYS': '#45B7AA', 'ARG': '#3DAA9E',
    'ASP': '#FFE66D', 'GLU': '#FFD93D',
    'PHE': '#C9B1FF', 'TYR': '#B8A0FF', 'TRP': '#A78BFA',
    'ALA': '#E8E8E8', 'VAL': '#E0E0E0', 'LEU': '#D8D8D8',
    'UNK': '#CCCCCC'
}

METAL_COLORS = {
    'ZN': '#5C7AEA', 'MG': '#06D6A0', 'FE': '#EF476F',
    'MN': '#9B5DE5', 'CU': '#F4A261', 'CA': '#2EC4B6',
    'DEFAULT': '#6C757D'
}

ROLE_COLORS = {
    'nucleophile': '#FF6B6B', 'general_base': '#4ECDC4',
    'general_acid': '#FFE66D', 'electrostatic': '#FFD93D',
    'metal_ligand': '#5C7AEA', 'other': '#888888'
}

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
