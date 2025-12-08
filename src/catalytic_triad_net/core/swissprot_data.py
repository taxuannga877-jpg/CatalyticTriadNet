#!/usr/bin/env python3
"""
Swiss-Prot数据获取和解析模块

Swiss-Prot是UniProtKB的人工审核部分，包含570,000+高质量蛋白质序列
相比M-CSA的~1,000条目，Swiss-Prot提供了更大规模的训练数据

数据来源：
- UniProt REST API: https://rest.uniprot.org/
- 数据量：570,000+ 蛋白质序列
- 酶数据：~200,000 条目
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwissProtEntry:
    """Swiss-Prot条目"""
    uniprot_id: str
    protein_name: str
    organism: str
    sequence: str
    ec_number: Optional[str]
    pdb_ids: List[str]
    active_sites: List[Dict]  # {'position': int, 'residue': str, 'description': str}
    binding_sites: List[Dict]
    metal_binding: List[Dict]
    catalytic_activity: Optional[str]
    function: Optional[str]


class SwissProtDataFetcher:
    """
    Swiss-Prot数据获取器

    使用UniProt REST API获取酶数据

    使用示例:
        fetcher = SwissProtDataFetcher()

        # 获取所有水解酶（EC 3）
        entries = fetcher.fetch_enzymes_by_ec_class(ec_class='3', limit=1000)

        # 获取特定EC号的酶
        entries = fetcher.fetch_enzymes_by_ec_number('3.4.21.4')

        # 获取有PDB结构的酶
        entries = fetcher.fetch_enzymes_with_structure(ec_class='3', limit=500)
    """

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(self, cache_dir: str = './data/swissprot_cache'):
        """
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Swiss-Prot数据获取器初始化: cache_dir={cache_dir}")

    def fetch_enzymes_by_ec_class(
        self,
        ec_class: str,
        limit: int = 1000,
        reviewed: bool = True,
        has_structure: bool = False
    ) -> List[Dict]:
        """
        按EC分类获取酶数据

        Args:
            ec_class: EC分类 ('1', '2', '3', '4', '5', '6', '7')
            limit: 最大条目数
            reviewed: 仅获取Swiss-Prot审核过的条目
            has_structure: 仅获取有PDB结构的条目

        Returns:
            条目列表
        """
        cache_file = self.cache_dir / f"ec{ec_class}_limit{limit}_struct{has_structure}.json"

        # 检查缓存
        if cache_file.exists():
            logger.info(f"从缓存加载: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # 构建查询
        query_parts = [f"(ec:{ec_class}.*)"]

        if reviewed:
            query_parts.append("(reviewed:true)")

        if has_structure:
            query_parts.append("(structure_3d:true)")

        query = " AND ".join(query_parts)

        logger.info(f"查询Swiss-Prot: EC {ec_class}, limit={limit}")
        logger.info(f"查询字符串: {query}")

        # 发送请求
        params = {
            'query': query,
            'format': 'json',
            'size': min(limit, 500),  # API限制每次最多500
            'fields': 'accession,id,protein_name,organism_name,sequence,ec,xref_pdb,ft_act_site,ft_binding,ft_metal,cc_catalytic_activity,cc_function'
        }

        entries = []
        cursor = None

        while len(entries) < limit:
            if cursor:
                params['cursor'] = cursor

            try:
                response = requests.get(
                    f"{self.BASE_URL}/search",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()

                data = response.json()
                results = data.get('results', [])

                if not results:
                    break

                entries.extend(results)
                logger.info(f"已获取 {len(entries)} 条目...")

                # 获取下一页的cursor
                cursor = response.headers.get('x-next-cursor')
                if not cursor:
                    break

                # 避免请求过快
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"请求失败: {e}")
                break

        # 保存缓存
        with open(cache_file, 'w') as f:
            json.dump(entries[:limit], f, indent=2)

        logger.info(f"✓ 获取完成: {len(entries[:limit])} 条目")

        return entries[:limit]

    def fetch_enzymes_by_ec_number(
        self,
        ec_number: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        按精确EC号获取酶数据

        Args:
            ec_number: 完整EC号 (如 '3.4.21.4')
            limit: 最大条目数

        Returns:
            条目列表
        """
        cache_file = self.cache_dir / f"ec_{ec_number.replace('.', '_')}.json"

        if cache_file.exists():
            logger.info(f"从缓存加载: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        query = f"(ec:{ec_number}) AND (reviewed:true)"

        params = {
            'query': query,
            'format': 'json',
            'size': limit,
            'fields': 'accession,id,protein_name,organism_name,sequence,ec,xref_pdb,ft_act_site,ft_binding,ft_metal,cc_catalytic_activity,cc_function'
        }

        try:
            response = requests.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            entries = data.get('results', [])

            # 保存缓存
            with open(cache_file, 'w') as f:
                json.dump(entries, f, indent=2)

            logger.info(f"✓ 获取 EC {ec_number}: {len(entries)} 条目")

            return entries

        except Exception as e:
            logger.error(f"请求失败: {e}")
            return []

    def fetch_enzymes_with_structure(
        self,
        ec_class: str,
        limit: int = 500
    ) -> List[Dict]:
        """
        获取有PDB结构的酶数据

        Args:
            ec_class: EC分类
            limit: 最大条目数

        Returns:
            条目列表
        """
        return self.fetch_enzymes_by_ec_class(
            ec_class=ec_class,
            limit=limit,
            reviewed=True,
            has_structure=True
        )

    def get_statistics(self, entries: List[Dict]) -> Dict:
        """
        统计Swiss-Prot数据

        Args:
            entries: 条目列表

        Returns:
            统计信息
        """
        stats = {
            'total_entries': len(entries),
            'with_pdb': 0,
            'with_active_site': 0,
            'with_binding_site': 0,
            'with_metal': 0,
            'ec_distribution': {},
            'organism_distribution': {}
        }

        for entry in entries:
            # PDB结构
            if entry.get('uniProtKBCrossReferences'):
                pdb_refs = [ref for ref in entry['uniProtKBCrossReferences']
                           if ref.get('database') == 'PDB']
                if pdb_refs:
                    stats['with_pdb'] += 1

            # 活性位点
            if entry.get('features'):
                for feat in entry['features']:
                    if feat.get('type') == 'Active site':
                        stats['with_active_site'] += 1
                        break

            # 结合位点
            if entry.get('features'):
                for feat in entry['features']:
                    if feat.get('type') == 'Binding site':
                        stats['with_binding_site'] += 1
                        break

            # 金属结合
            if entry.get('features'):
                for feat in entry['features']:
                    if feat.get('type') == 'Metal binding':
                        stats['with_metal'] += 1
                        break

            # EC分布
            if entry.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers'):
                ec_numbers = entry['proteinDescription']['recommendedName']['ecNumbers']
                for ec in ec_numbers:
                    ec_value = ec.get('value', '')
                    ec_class = ec_value.split('.')[0] if ec_value else 'unknown'
                    stats['ec_distribution'][ec_class] = stats['ec_distribution'].get(ec_class, 0) + 1

            # 物种分布
            organism = entry.get('organism', {}).get('scientificName', 'unknown')
            stats['organism_distribution'][organism] = stats['organism_distribution'].get(organism, 0) + 1

        return stats


class SwissProtDataParser:
    """
    Swiss-Prot数据解析器

    将Swiss-Prot JSON数据解析为结构化对象
    """

    @staticmethod
    def parse_entry(entry: Dict) -> SwissProtEntry:
        """
        解析单个Swiss-Prot条目

        Args:
            entry: Swiss-Prot JSON条目

        Returns:
            SwissProtEntry对象
        """
        # 基本信息
        uniprot_id = entry.get('primaryAccession', '')
        protein_name = entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        organism = entry.get('organism', {}).get('scientificName', '')
        sequence = entry.get('sequence', {}).get('value', '')

        # EC号
        ec_number = None
        if entry.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers'):
            ec_numbers = entry['proteinDescription']['recommendedName']['ecNumbers']
            if ec_numbers:
                ec_number = ec_numbers[0].get('value')

        # PDB IDs
        pdb_ids = []
        if entry.get('uniProtKBCrossReferences'):
            for ref in entry['uniProtKBCrossReferences']:
                if ref.get('database') == 'PDB':
                    pdb_ids.append(ref.get('id'))

        # 活性位点
        active_sites = []
        binding_sites = []
        metal_binding = []

        if entry.get('features'):
            for feat in entry['features']:
                feat_type = feat.get('type')
                location = feat.get('location', {})
                start = location.get('start', {}).get('value')
                description = feat.get('description', '')

                if feat_type == 'Active site':
                    active_sites.append({
                        'position': start,
                        'description': description
                    })
                elif feat_type == 'Binding site':
                    binding_sites.append({
                        'position': start,
                        'description': description
                    })
                elif feat_type == 'Metal binding':
                    metal_binding.append({
                        'position': start,
                        'description': description
                    })

        # 催化活性
        catalytic_activity = None
        if entry.get('comments'):
            for comment in entry['comments']:
                if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                    reaction = comment.get('reaction', {})
                    catalytic_activity = reaction.get('name', '')
                    break

        # 功能
        function = None
        if entry.get('comments'):
            for comment in entry['comments']:
                if comment.get('commentType') == 'FUNCTION':
                    texts = comment.get('texts', [])
                    if texts:
                        function = texts[0].get('value', '')
                    break

        return SwissProtEntry(
            uniprot_id=uniprot_id,
            protein_name=protein_name,
            organism=organism,
            sequence=sequence,
            ec_number=ec_number,
            pdb_ids=pdb_ids,
            active_sites=active_sites,
            binding_sites=binding_sites,
            metal_binding=metal_binding,
            catalytic_activity=catalytic_activity,
            function=function
        )

    @staticmethod
    def parse_entries(entries: List[Dict]) -> List[SwissProtEntry]:
        """
        批量解析Swiss-Prot条目

        Args:
            entries: Swiss-Prot JSON条目列表

        Returns:
            SwissProtEntry对象列表
        """
        parsed_entries = []

        for entry in entries:
            try:
                parsed = SwissProtDataParser.parse_entry(entry)
                parsed_entries.append(parsed)
            except Exception as e:
                logger.warning(f"解析条目失败: {e}")

        return parsed_entries


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'SwissProtEntry',
    'SwissProtDataFetcher',
    'SwissProtDataParser',
]


if __name__ == "__main__":
    # 测试代码
    print("Swiss-Prot数据获取器 - 测试")
    print("="*60)

    fetcher = SwissProtDataFetcher()

    # 获取水解酶数据
    print("\n获取水解酶（EC 3）数据...")
    entries = fetcher.fetch_enzymes_by_ec_class(ec_class='3', limit=100)

    print(f"\n✓ 获取了 {len(entries)} 个条目")

    # 统计
    stats = fetcher.get_statistics(entries)
    print(f"\n统计信息:")
    print(f"  总条目数: {stats['total_entries']}")
    print(f"  有PDB结构: {stats['with_pdb']}")
    print(f"  有活性位点标注: {stats['with_active_site']}")
    print(f"  有结合位点标注: {stats['with_binding_site']}")
    print(f"  有金属结合标注: {stats['with_metal']}")

    # 解析第一个条目
    if entries:
        print(f"\n解析第一个条目...")
        parser = SwissProtDataParser()
        parsed = parser.parse_entry(entries[0])

        print(f"\nUniProt ID: {parsed.uniprot_id}")
        print(f"蛋白质名称: {parsed.protein_name}")
        print(f"EC号: {parsed.ec_number}")
        print(f"PDB IDs: {parsed.pdb_ids}")
        print(f"活性位点数: {len(parsed.active_sites)}")
        print(f"序列长度: {len(parsed.sequence)}")
