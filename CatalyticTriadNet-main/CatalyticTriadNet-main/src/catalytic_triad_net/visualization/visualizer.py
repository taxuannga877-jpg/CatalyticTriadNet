#!/usr/bin/env python3
"""
主可视化器 - 整合所有可视化功能

提供统一的可视化接口，支持2D/3D可视化、专业软件导出和扩散模型适配。
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import numpy as np
import os

from .adapters import DiffusionModelAdapter
from .exporters import ProfessionalExporter
from .plot_2d import Visualizer2D
from .plot_3d import Visualizer3D

logger = logging.getLogger(__name__)

# 检查Plotly是否可用
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.debug("Plotly not available, interactive visualizations will be disabled")

# 检查matplotlib是否可用
try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True

    # 检测是否在无显示环境中
    if os.environ.get('DISPLAY', '') == '':
        logger.info("No display detected, using Agg backend for matplotlib")
        matplotlib.use('Agg')
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, 2D/3D visualizations will be disabled")


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

    def __init__(self, figsize_2d: tuple = (12, 10), figsize_3d: tuple = (12, 10)):
        """
        初始化可视化器。

        Args:
            figsize_2d: 2D图形尺寸
            figsize_3d: 3D图形尺寸
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, visualization features will be limited")
            self.viz2d = None
            self.viz3d = None
        else:
            self.viz2d = Visualizer2D(figsize_2d)
            self.viz3d = Visualizer3D(figsize_3d)

        self.adapter = DiffusionModelAdapter()
        self.exporter = ProfessionalExporter()

    def visualize(
        self,
        results: Dict,
        coords: Optional[np.ndarray] = None,
        output_dir: Union[str, Path] = "./output",
        prefix: str = "",
        modes: Optional[List[str]] = None
    ) -> None:
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

        logger.info(f"{'='*50}")
        logger.info(f"可视化: {prefix}")
        logger.info(f"{'='*50}")

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping 2D/3D visualizations")
            modes = [m for m in modes if m == 'interactive' and PLOTLY_AVAILABLE]

        try:
            if '2d_graph' in modes and self.viz2d:
                self.viz2d.plot_molecular_graph(results, output_dir / f"{prefix}_2d_graph.png")

            if '2d_triad' in modes and self.viz2d:
                self.viz2d.plot_triads(results, output_dir / f"{prefix}_2d_triads.png")

            if '2d_metal' in modes and self.viz2d:
                self.viz2d.plot_metal_centers(results, output_dir / f"{prefix}_2d_metals.png")

            if '3d_site' in modes and self.viz3d:
                self.viz3d.plot_active_site(results, coords, output_dir / f"{prefix}_3d_site.png")

            if '3d_metal' in modes and self.viz3d:
                self.viz3d.plot_metal_polyhedra(results, coords, output_dir / f"{prefix}_3d_polyhedra.png")

            if 'interactive' in modes and PLOTLY_AVAILABLE and self.viz3d:
                self.viz3d.plot_interactive(results, coords, output_dir / f"{prefix}_interactive.html")
            elif 'interactive' in modes and not PLOTLY_AVAILABLE:
                logger.warning("Plotly not available, skipping interactive visualization")

            if 'rotation' in modes and self.viz3d:
                self.viz3d.export_rotation_gif(results, coords, output_dir / f"{prefix}_rotation.gif")

            logger.info(f"所有可视化已保存到: {output_dir}")

        except Exception as e:
            logger.error(f"可视化过程中出错: {e}", exc_info=True)
            raise

    def visualize_diffusion(
        self,
        node_types: np.ndarray,
        edge_index: np.ndarray,
        coords: Optional[np.ndarray] = None,
        edge_types: Optional[np.ndarray] = None,
        atom_list: Optional[List[str]] = None,
        output_dir: Union[str, Path] = "./output",
        name: str = "generated",
        **kwargs
    ) -> Dict:
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
            **kwargs: 其他参数

        Returns:
            适配后的结果字典
        """
        try:
            results = self.adapter.from_graph_data(
                node_types, edge_index, coords, edge_types, atom_list, name=name, **kwargs
            )

            self.visualize(
                results, results.get('coords'), output_dir, name,
                modes=['2d_graph', '3d_site', 'interactive']
            )

            return results

        except Exception as e:
            logger.error(f"扩散模型可视化出错: {e}", exc_info=True)
            raise

    def visualize_pyg(
        self,
        data,
        atom_list: Optional[List[str]] = None,
        output_dir: Union[str, Path] = "./output",
        name: str = "pyg_mol"
    ) -> Dict:
        """
        可视化PyG Data对象

        Args:
            data: PyG Data对象
            atom_list: 原子类型映射
            output_dir: 输出目录
            name: 结构名称

        Returns:
            适配后的结果字典
        """
        try:
            results = self.adapter.from_pyg_data(data, atom_list)
            self.visualize(
                results, results.get('coords'), output_dir, name,
                modes=['2d_graph', '3d_site']
            )
            return results

        except Exception as e:
            logger.error(f"PyG数据可视化出错: {e}", exc_info=True)
            raise

    def visualize_rfdiffusion(
        self,
        output_path: Union[str, Path],
        output_dir: Union[str, Path] = "./output"
    ) -> Dict:
        """
        可视化RFdiffusion输出

        Args:
            output_path: RFdiffusion输出文件路径
            output_dir: 输出目录

        Returns:
            适配后的结果字典
        """
        try:
            results = self.adapter.from_rfdiffusion(str(output_path))
            self.visualize(results, results.get('coords'), output_dir, results.get('pdb_id'))
            return results

        except Exception as e:
            logger.error(f"RFdiffusion输出可视化出错: {e}", exc_info=True)
            raise

    def visualize_nanozyme_design(
        self,
        design_path: Union[str, Path],
        output_dir: Union[str, Path] = "./output"
    ) -> Dict:
        """
        可视化纳米酶设计输入

        Args:
            design_path: 纳米酶设计文件路径
            output_dir: 输出目录

        Returns:
            适配后的结果字典
        """
        try:
            results = self.adapter.from_nanozyme_design(str(design_path))
            self.visualize(
                results, None, output_dir, results.get('pdb_id'),
                modes=['2d_triad', '2d_metal', '3d_metal']
            )
            return results

        except Exception as e:
            logger.error(f"纳米酶设计可视化出错: {e}", exc_info=True)
            raise

    def export_professional(
        self,
        results: Dict,
        pdb_path: Union[str, Path],
        output_dir: Union[str, Path] = "./output",
        prefix: Optional[str] = None
    ) -> None:
        """
        导出专业软件脚本

        Args:
            results: 预测结果字典
            pdb_path: PDB文件路径
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = prefix or results.get('pdb_id', 'output')

        try:
            logger.info("导出专业软件脚本...")
            self.exporter.to_pymol(results, str(pdb_path), str(output_dir / f"{prefix}.pml"))
            self.exporter.to_chimerax(results, str(pdb_path), str(output_dir / f"{prefix}.cxc"))
            self.exporter.to_vmd(results, str(pdb_path), str(output_dir / f"{prefix}.tcl"))

            # 如果是生成的小分子，导出简化PDB
            if results.get('_is_molecule') and results.get('coords') is not None:
                self.exporter.to_pdb(
                    results, results['coords'],
                    str(output_dir / f"{prefix}_generated.pdb"),
                    results.get('node_symbols')
                )

            logger.info(f"专业软件脚本已保存到: {output_dir}")

        except Exception as e:
            logger.error(f"导出专业软件脚本出错: {e}", exc_info=True)
            raise


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
    print("演示完成！所有文件保存在 ./demo_output/")
    print("="*60)


if __name__ == "__main__":
    demo()
