#!/usr/bin/env python3
"""
预测示例 - 使用重构后的模块
"""

from catalytic_triad_net import EnhancedCatalyticSiteInference

# 初始化预测器
predictor = EnhancedCatalyticSiteInference(
    model_path='models/best_model.pt',
    device='cuda'  # 或 'cpu'
)

# 预测 (支持PDB ID或本地文件)
results = predictor.predict(
    pdb_path='1acb',  # 丝氨酸蛋白酶
    site_threshold=0.5
)

# 打印结果
predictor.print_results(results, top_k=15)

# 导出格式
predictor.export_pymol(results, 'output/1acb.pml')
predictor.export_for_proteinmpnn(results, 'output/1acb_mpnn.json')
predictor.export_for_rfdiffusion(results, 'output/1acb_rfd.json')

print("\n✓ 预测完成!")
