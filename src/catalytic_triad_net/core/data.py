#!/usr/bin/env python3
"""
M-CSA数据获取和处理模块
"""

import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# M-CSA API数据获取
# =============================================================================

class MCSADataFetcher:
    """M-CSA API数据获取器"""

    BASE_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api"

    def __init__(self, cache_dir: str = "./data/mcsa_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all_entries(self, force_refresh: bool = False) -> List[Dict]:
        """获取所有M-CSA entries"""
        cache_file = self.cache_dir / "mcsa_entries_full.json"

        if cache_file.exists() and not force_refresh:
            logger.info(f"从缓存加载: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        logger.info("从M-CSA API获取entries...")
        url = f"{self.BASE_URL}/entries/?format=json"
        all_results = []

        while url:
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                data = r.json()
                all_results.extend(data["results"])
                url = data.get("next")
                logger.info(f"已获取 {len(all_results)}/{data['count']} entries")
            except Exception as e:
                logger.error(f"获取失败: {e}")
                break

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        return all_results

    def fetch_all_residues(self, force_refresh: bool = False) -> List[Dict]:
        """获取所有催化残基"""
        cache_file = self.cache_dir / "mcsa_residues_full.json"

        if cache_file.exists() and not force_refresh:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        logger.info("从M-CSA API获取residues...")
        url = f"{self.BASE_URL}/residues/?format=json"
        all_results = []

        while url:
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                data = r.json()
                all_results.extend(data["results"])
                url = data.get("next")
            except Exception as e:
                logger.error(f"获取失败: {e}")
                break

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        return all_results


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class CatalyticResidue:
    """催化残基"""
    residue_name: str
    residue_number: int
    chain_id: str
    pdb_id: str
    roles: List[str] = field(default_factory=list)


@dataclass
class EnzymeEntry:
    """酶条目"""
    mcsa_id: str
    enzyme_name: str
    ec_numbers: List[str]
    pdb_id: str
    catalytic_residues: List[CatalyticResidue] = field(default_factory=list)


# =============================================================================
# M-CSA数据解析
# =============================================================================

class MCSADataParser:
    """M-CSA数据解析器"""

    ROLE_MAPPING = {
        'nucleophile': 'nucleophile',
        'proton donor': 'proton_donor',
        'proton acceptor': 'proton_acceptor',
        'electrostatic stabiliser': 'electrostatic_stabilizer',
        'metal ligand': 'metal_binding',
        'covalent catalyst': 'covalent_catalyst',
        'activator': 'activator',
        'steric role': 'steric_role',
    }

    def parse_entries(self, entries_json: List[Dict]) -> List[EnzymeEntry]:
        """解析entries"""
        parsed = []
        for entry in tqdm(entries_json, desc="解析M-CSA"):
            try:
                enzyme = self._parse_entry(entry)
                if enzyme and enzyme.catalytic_residues:
                    parsed.append(enzyme)
            except Exception as e:
                logger.debug(f"解析失败: {e}")

        logger.info(f"✓ 解析 {len(parsed)} 个酶条目")
        return parsed

    def _parse_entry(self, entry: Dict) -> Optional[EnzymeEntry]:
        """解析单个entry"""
        residues_data = entry.get('residues', [])
        if not residues_data:
            return None

        chains = residues_data[0].get('residue_chains', [])
        if not chains:
            return None

        pdb_id = chains[0].get('pdb_id', '').lower()
        if not pdb_id:
            return None

        catalytic_residues = []
        for res in residues_data:
            cat_res = self._parse_residue(res, pdb_id)
            if cat_res:
                catalytic_residues.append(cat_res)

        return EnzymeEntry(
            mcsa_id=entry.get('mcsa_id', ''),
            enzyme_name=entry.get('enzyme_name', ''),
            ec_numbers=entry.get('all_ecs', []),
            pdb_id=pdb_id,
            catalytic_residues=catalytic_residues
        )

    def _parse_residue(self, res_data: Dict, pdb_id: str) -> Optional[CatalyticResidue]:
        """解析催化残基"""
        chains = res_data.get('residue_chains', [])
        if not chains:
            return None

        c = chains[0]
        roles = []
        for role in res_data.get('roles', []):
            func = role.get('function', '').lower()
            for k, v in self.ROLE_MAPPING.items():
                if k in func:
                    roles.append(v)
                    break

        return CatalyticResidue(
            residue_name=c.get('code', ''),
            residue_number=c.get('resid', 0),
            chain_id=c.get('chain_name', 'A'),
            pdb_id=pdb_id,
            roles=roles or ['other']
        )

    def get_statistics(self, entries: List[EnzymeEntry]) -> Dict:
        """统计信息"""
        stats = {
            'total_entries': len(entries),
            'total_catalytic': sum(len(e.catalytic_residues) for e in entries),
            'unique_pdbs': len(set(e.pdb_id for e in entries)),
            'ec_dist': defaultdict(int),
            'res_dist': defaultdict(int),
        }

        for e in entries:
            for ec in e.ec_numbers:
                ec1 = ec.split('.')[0] if '.' in ec else ec
                stats['ec_dist'][ec1] += 1
            for r in e.catalytic_residues:
                stats['res_dist'][r.residue_name] += 1

        return stats
