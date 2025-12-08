#!/usr/bin/env python3
"""
专业软件导出器
支持: PyMOL, ChimeraX, VMD
"""

from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

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
