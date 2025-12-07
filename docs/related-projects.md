# 与你课题高度相关的GitHub高星项目

根据你的代码`CatalyticDiff`（基于催化位点几何约束的条件分子扩散生成模型），我找到了以下高度相关的项目：

---

## 🔥 核心技术高度契合（E(3)-等变扩散 + 条件生成）

### 1. EDM - E(3) Equivariant Diffusion Model
**GitHub**: https://github.com/ehoogeboom/e3_diffusion_for_molecules  
**⭐ 契合度: ★★★★★**

你的代码直接参考了这个项目。EDM是3D分子E(3)等变扩散模型的开山之作：
- E(3)等变EGNN网络架构（你的`E3EquivariantLayer`直接借鉴）
- DDPM扩散框架
- 支持条件生成（基于分子性质如α, gap, homo等）
- QM9和GEOM-Drugs数据集

### 2. DiffLinker - 分子连接器设计
**GitHub**: https://github.com/igashov/DiffLinker  
**⭐ 契合度: ★★★★★**

与你的"基于约束的条件生成"思路非常接近：
- **条件化**: 给定多个片段的3D位置，生成连接它们的分子
- **几何约束**: 必须精确连接指定的锚点原子
- **蛋白口袋条件化**: 可以在口袋内生成linker
- E(3)等变扩散模型

### 3. TargetDiff - 靶向分子生成
**GitHub**: https://github.com/guanjq/targetdiff  
**⭐ 契合度: ★★★★★**

你的代码也引用了这个项目，核心相似点：
- **条件化生成**: 以蛋白口袋3D结构为条件
- **E(3)等变**: 保证旋转平移不变性
- **几何约束**: 生成适配口袋形状的分子
- ICLR 2023

### 4. GeoLDM - 几何潜在扩散模型
**GitHub**: https://github.com/MinkaiXu/GeoLDM  
**⭐ 契合度: ★★★★☆**

你代码中提到的参考项目：
- **潜在空间扩散**: 在压缩表示空间做扩散（更高效）
- **SE(3)等变自编码器**: 保持几何约束
- 更好的可控生成能力
- ICML 2023

### 5. PMDM - 双扩散模型
**GitHub**: https://github.com/Layne-Huang/PMDM  
**⭐ 契合度: ★★★★☆**

你代码引用的项目：
- **双扩散框架**: 同时建模局部和全局分子动力学
- **蛋白条件化**: 语义信息+空间信息双重条件
- **先导化合物优化**: 支持从种子片段生成
- Nature Communications 2024

---

## 🎯 催化位点/酶设计方向（最相关！）

### 6. RFdiffusion2 🆕
**GitHub**: https://github.com/RosettaCommons/RFdiffusion2  
**⭐ 契合度: ★★★★★ (最推荐！)**

**与你课题最匹配的项目！** 刚发布的Baker Lab新工作：
- **原子级催化位点scaffolding**: 从功能基团几何直接设计酶
- **不需预定义残基位置**: 网络自动推断序列索引和旋转异构体
- **催化三联体设计**: 支持Ser-His-Asp等经典催化三联体
- **theozyme条件化**: 以过渡态周围的催化基团配置为输入
- Nature Methods 2025

### 7. RFdiffusion (原版)
**GitHub**: https://github.com/RosettaCommons/RFdiffusion  
**⭐ 契合度: ★★★★☆**

经典的蛋白质扩散生成模型：
- **Motif scaffolding**: 给定功能motif，生成包含该motif的蛋白
- **酶活性位点设计**: 可以指定催化残基位置
- 支持金属结合位点设计

### 8. Riff-Diff (催化motif scaffolding)
**参考**: Nature 2025刚发表
**⭐ 契合度: ★★★★★**

与RFdiffusion2配套的工作：
- **催化阵列scaffolding**: 从催化残基阵列生成酶
- **retro-aldol和Morita-Baylis-Hillman反应**: 实验验证
- **高精度活性位点**: 埃级活性位点设计精度

---

## 🔬 结构药物设计（蛋白口袋条件生成）

### 9. DiffSBDD
**GitHub**: https://github.com/arneschneuing/DiffSBDD  
**⭐ 契合度: ★★★★☆**

结构药物设计等变扩散模型：
- **口袋条件化**: 以蛋白口袋为条件生成配体
- **inpainting**: 部分分子重设计
- **优化功能**: 基于QED/SA等性质优化

### 10. DecompDiff
**GitHub**: https://github.com/bytedance/DecompDiff  
**⭐ 契合度: ★★★☆☆**

字节跳动的分解先验扩散模型：
- **功能基团分解**: 将分子分解为功能基团
- **ICML 2023**

### 11. NucleusDiff
**GitHub**: https://github.com/yanliang3612/NucleusDiff  
**⭐ 契合度: ★★★☆☆**

流形约束的核级扩散模型：
- **流形约束**: 保证生成分子在化学有效流形上
- **PNAS 2025**

---

## 📚 综合资源库

### 12. Awesome-SBDD
**GitHub**: https://github.com/zaixizhang/Awesome-SBDD  
**精选的结构药物设计论文列表，涵盖所有主要方法**

### 13. Awesome Molecular Diffusion Models
**GitHub**: https://github.com/AzureLeon1/awesome-molecular-diffusion-models  
**分子扩散模型综合列表**

### 14. Papers for Molecular Design using DL
**GitHub**: https://github.com/AspirinCode/papers-for-molecular-design-using-DL  
**最全面的分子设计深度学习论文列表**

---

## 🔗 与你课题的具体对应关系

| 你的代码模块 | 最相关项目 | 借鉴点 |
|-------------|-----------|--------|
| `E3EquivariantLayer` | EDM, TargetDiff | EGNN架构、消息传递 |
| `CatalyticConstraints` | **RFdiffusion2** | 原子级motif条件化 |
| `ConditionEncoder` | DiffLinker, PMDM | 条件融合机制 |
| 距离/角度约束 | DiffLinker | 几何约束损失 |
| 金属配位中心 | RFdiffusion | 金属结合位点设计 |
| 催化三联体 | **RFdiffusion2, Riff-Diff** | 催化阵列scaffolding |

---

## 💡 建议优先关注

1. **RFdiffusion2** - 与你的"催化位点几何约束"思想最匹配
2. **DiffLinker** - 条件化几何约束生成的优秀实现
3. **EDM** - E(3)等变扩散的基础代码
4. **TargetDiff** - 蛋白口袋条件化的参考

你的CatalyticDiff独特之处在于：
- **专注纳米酶** (vs 蛋白/小分子药物)
- **金属配位环境建模** (Fe, Cu, Zn, Ce等)
- **催化三联体几何约束** 作为核心条件
