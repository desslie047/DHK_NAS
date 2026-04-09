import os.path as osp
import numpy as np
from collections import Counter
import torch


def analyze_crossdomain_dataset():
    """分析CrossDomain数据集的特征"""

    fnames = ["Database", "Data Mining", "Medical Informatics", "Theory", "Visualization"]
    dataroot = osp.join("data", "Cross-Domain_data")
    datafiles = [f"{dataroot}/{name}.txt" for name in fnames]

    all_stats = []

    for fname, datafile in zip(fnames, datafiles):
        if not osp.exists(datafile):
            print(f"文件不存在: {datafile}")
            continue

        with open(datafile, "r") as file:
            lines = file.readlines()

        papers = []
        venues = set()
        authors = set()
        years = []

        for line in lines:
            try:
                venue, title, author_str, year, abstract = line.split("\t")
                year = int(year)

                venues.add(venue)
                years.append(year)
                for author in author_str.split(","):
                    authors.add(author.strip())
                papers.append((venue, title, author_str, year, abstract))
            except:
                continue

        # 统计边数量
        num_pa_edges = sum(len(p[2].split(",")) for p in papers)  # paper-author edges
        num_pv_edges = len(papers)  # paper-venue edges

        stats = {
            'dataset_name': fname,
            'num_nodes': {
                'paper': len(papers),
                'author': len(authors),
                'venue': len(venues),
                'total': len(papers) + len(authors) + len(venues)
            },
            'num_edges': {
                'paper-author': num_pa_edges,
                'paper-venue': num_pv_edges,
                'total': num_pa_edges + num_pv_edges
            },
            'node_types': ['paper', 'author', 'venue'],
            'edge_types': ['written (paper-author)', 'published (paper-venue)'],
            'time_span': {
                'start': min(years) if years else None,
                'end': max(years) if years else None,
                'range': f"{min(years)}-{max(years)}" if years else None
            }
        }

        all_stats.append(stats)

    return all_stats


def analyze_merged_dataset():
    """分析合并后的完整数据集"""
    from dhknas.data.crossdomain import CrossDomainDataset

    dataset = CrossDomainDataset(undirected=False)
    data = dataset.dataset

    # 统计节点数量
    num_papers = data['paper'].num_nodes
    num_authors = data['author'].num_nodes
    num_venues = data['venue'].num_nodes

    # 统计边数量
    num_pa_edges = data['paper', 'written', 'author'].edge_index.shape[1]
    num_pv_edges = data['paper', 'published', 'venue'].edge_index.shape[1]

    # 时间跨度
    years = data['paper'].time.squeeze().unique().sort()[0].numpy()

    stats = {
        'dataset_name': 'CrossDomain (合并)',
        'num_nodes': {
            'paper': num_papers,
            'author': num_authors,
            'venue': num_venues,
            'total': num_papers + num_authors + num_venues
        },
        'num_edges': {
            'written (paper-author)': num_pa_edges,
            'published (paper-venue)': num_pv_edges,
            'total': num_pa_edges + num_pv_edges
        },
        'node_types': ['paper', 'author', 'venue'],
        'edge_types': ['written', 'published'],
        'time_span': {
            'time_steps': len(years),
            'years': years.tolist(),
            'range': f"{years[0]}-{years[-1]}"
        }
    }

    return stats


def print_statistics(stats_list):
    """打印统计结果"""
    for stats in stats_list:
        print(f"\n{'=' * 70}")
        print(f"数据集: {stats['dataset_name']}")
        print(f"\n节点数量:")
        for node_type, count in stats['num_nodes'].items():
            print(f"  {node_type}: {count}")

        print(f"\n边数量:")
        for edge_type, count in stats['num_edges'].items():
            print(f"  {edge_type}: {count}")

        print(f"\n节点类型: {stats['node_types']}")
        print(f"边类型: {stats['edge_types']}")

        print(f"\n时间跨度:")
        for key, value in stats['time_span'].items():
            print(f"  {key}: {value}")
        print(f"{'=' * 70}")


# 分析单个领域数据集
individual_stats = analyze_crossdomain_dataset()
print_statistics(individual_stats)

# 分析合并后的数据集
merged_stats = analyze_merged_dataset()
print_statistics([merged_stats])
