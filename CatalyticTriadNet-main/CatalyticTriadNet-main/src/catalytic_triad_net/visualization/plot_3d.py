#!/usr/bin/env python3
"""
3D可视化模块
包含: 空间分布、配位多面体、交互式Plotly图表
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 颜色定义
ROLE_COLORS = {
    'nucleophile': '#FF6B6B',
    'general_base': '#4ECDC4',
    'general_acid': '#FFE66D',
    'metal_ligand': '#95E1D3',
    'transition_state_stabilizer': '#F38181',
    'proton_donor': '#AA96DA',
    'proton_acceptor': '#FCBAD3',
    'electrostatic_stabilizer': '#A8E6CF',
    'other': '#CCCCCC'
}

AA_COLORS = {
    'ALA': '#C8C8C8', 'ARG': '#145AFF', 'ASN': '#00DCDC', 'ASP': '#E60A0A',
    'CYS': '#E6E600', 'GLN': '#00DCDC', 'GLU': '#E60A0A', 'GLY': '#EBEBEB',
    'HIS': '#8282D2', 'ILE': '#0F820F', 'LEU': '#0F820F', 'LYS': '#145AFF',
    'MET': '#E6E600', 'PHE': '#3232AA', 'PRO': '#DC9682', 'SER': '#FA9600',
    'THR': '#FA9600', 'TRP': '#B45AB4', 'TYR': '#3232AA', 'VAL': '#0F820F',
    'UNK': '#CCCCCC'
}

METAL_COLORS = {
    'FE': '#E06633', 'CU': '#C88033', 'ZN': '#7D80B0', 'MG': '#8AFF00',
    'CA': '#3DFF00', 'MN': '#9C7AC7', 'CO': '#F090A0', 'NI': '#50D050',
    'MO': '#54B5B5', 'W': '#2194D6', 'V': '#A6A6AB', 'CE': '#FFFFC7',
    'PT': '#D0D0E0', 'PD': '#006985', 'AU': '#FFD123', 'AG': '#C0C0C0',
    'DEFAULT': '#FF1493'
}

# 配位几何模板
COORD_GEOMETRIES = {
    'tetrahedral': {
        'v': np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])/np.sqrt(3),
        'f': [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
    },
    'octahedral': {
        'v': np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]),
        'f': [[0,2,4],[0,4,3],[0,3,5],[0,5,2],[1,2,4],[1,4,3],[1,3,5],[1,5,2]]
    },
    'square_planar': {
        'v': np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]])/np.sqrt(2),
        'f': [[0,1,2,3]]
    }
}

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
