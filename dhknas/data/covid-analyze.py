import os.path as osp
import torch
from collections import Counter
from dhknas import config


def analyze_covid_raw_data():
    """分析COVID原始数据"""
    from dgl.data.utils import load_graphs

    dataroot = osp.join(config.dataroot, "covid/")
    glist, _ = load_graphs(osp.join(dataroot, "covid_graphs.bin"))

    print(f"原始图数量: {len(glist)}")

    # 分析第一个图的结构
    g = glist[0]

    node_stats = {}
    for ntype in g.ntypes:
        node_stats[ntype] = g.num_nodes(ntype)

    edge_stats = {}
    for stype, etype, ttype in g.canonical_etypes:
        edge_key = f"{stype}-{etype}-{ttype}"
        edge_stats[edge_key] = g.num_edges(etype)

    # 特征维度
    feature_dims = {}
    for ntype in g.ntypes:
        if 'feat' in g.nodes[ntype].data:
            feature_dims[ntype] = g.nodes[ntype].data['feat'].shape[1]

    stats = {
        'dataset_name': 'COVID (原始)',
        'num_graphs': len(glist),
        'num_nodes': node_stats,
        'num_edges': edge_stats,
        'node_types': g.ntypes,
        'edge_types': [etype for _, etype, _ in g.canonical_etypes],
        'feature_dims': feature_dims,
        'time_span': {
            'time_steps': len(glist),
            'range': f"0 - {len(glist) - 1}",
            'description': '每个图对应一个时间步'
        }
    }

    return stats


def analyze_covid_processed_dataset():
    """分析处理后的COVID数据集"""
    from dhknas.data.covid import CovidDataset

    dataset = CovidDataset()
    datas = dataset.dataset
    times = dataset.times()

    # 分析第一个时间步的数据
    data = datas[0]

    # 统计节点数量
    node_stats = {}
    node_types = []
    feature_dims = {}
    for ntype in data.node_types:
        node_stats[ntype] = data[ntype].num_nodes
        node_types.append(ntype)
        if hasattr(data[ntype], 'x'):
            feature_dims[ntype] = data[ntype].x.shape[1]
    node_stats['total'] = sum(node_stats.values())

    # 统计边数量
    edge_stats = {}
    edge_types = []
    for edge_type in data.edge_types:
        edge_count = data[edge_type].edge_index.shape[1]
        edge_key = f"{edge_type[0]}-{edge_type[2]}-{edge_type[1]}"
        edge_stats[edge_key] = edge_count
        edge_types.append(edge_type[2])
    edge_stats['total'] = sum(edge_stats.values())

    stats = {
        'dataset_name': 'COVID (处理后)',
        'num_graphs': len(datas),
        'num_nodes': node_stats,
        'num_edges': edge_stats,
        'node_types': node_types,
        'edge_types': list(set(edge_types)),
        'feature_dims': feature_dims,
        'time_span': {
            'time_steps': len(times),
            'times': times,
            'range': f"{times[0]} - {times[-1]}",
            'duration': f"{len(times)} 个时间步",
            'description': '2020年疫情数据快照'
        }
    }

    return stats


def analyze_covid_uni_dataset():
    """分析COVID统一格式数据集（用于训练）"""
    from dhknas.data.covid import CovidUniDataset

    # 不同时间窗口配置
    configs = [
        {'time_window': 1, 'is_dynamic': False},
        {'time_window': 3, 'is_dynamic': True},
        {'time_window': 7, 'is_dynamic': True},
    ]

    all_stats = []

    for cfg in configs:
        dataset = CovidUniDataset(**cfg, shuffle=False)

        # 统计数据划分
        split_stats = {}
        for split in ['train', 'val', 'test']:
            split_data = dataset.time_dataset[split]
            split_stats[split] = len(split_data)

        stats = {
            'dataset_name': f"COVID (time_window={cfg['time_window']}, dynamic={cfg['is_dynamic']})",
            'time_window': cfg['time_window'],
            'is_dynamic': cfg['is_dynamic'],
            'splits': split_stats,
            'total_samples': sum(split_stats.values()),
            'metadata': {
                'node_types': dataset.metadata[0],
                'edge_types': dataset.metadata[1]
            },
            'time_span': {
                'total_time_steps': len(dataset.dataset),
                'train_time_range': f"time_window - {len(dataset.dataset) - 60}",
                'val_time_range': f"{len(dataset.dataset) - 60} - {len(dataset.dataset) - 30}",
                'test_time_range': f"{len(dataset.dataset) - 30} - {len(dataset.dataset) - 1}",
                'description': '最后30个时间步用于测试，之前30个用于验证'
            }
        }

        all_stats.append(stats)

    return all_stats


def print_statistics(stats):
    """打印统计结果"""
    if isinstance(stats, list):
        for s in stats:
            print_single_statistics(s)
    else:
        print_single_statistics(stats)


def print_single_statistics(stats):
    """打印单个统计结果"""
    print(f"\n{'=' * 70}")
    print(f"数据集: {stats['dataset_name']}")

    if 'num_graphs' in stats:
        print(f"\n图数量: {stats['num_graphs']}")

    if 'num_nodes' in stats:
        print(f"\n节点数量:")
        for node_type, count in stats['num_nodes'].items():
            print(f"  {node_type}: {count}")

    if 'num_edges' in stats:
        print(f"\n边数量:")
        for edge_type, count in stats['num_edges'].items():
            print(f"  {edge_type}: {count}")

    if 'node_types' in stats:
        print(f"\n节点类型: {stats['node_types']}")

    if 'edge_types' in stats:
        print(f"\n边类型: {stats['edge_types']}")

    if 'feature_dims' in stats:
        print(f"\n特征维度:")
        for ntype, dim in stats['feature_dims'].items():
            print(f"  {ntype}: {dim}")

    if 'time_window' in stats:
        print(f"\n时间窗口: {stats['time_window']}")
        print(f"动态模式: {stats['is_dynamic']}")

    if 'splits' in stats:
        print(f"\n数据划分:")
        for split, count in stats['splits'].items():
            print(f"  {split}: {count}")
        print(f"  total: {stats['total_samples']}")

    if 'metadata' in stats and 'node_types' in stats['metadata']:
        print(f"\n元数据:")
        print(f"  节点类型: {stats['metadata']['node_types']}")
        print(f"  边类型: {stats['metadata']['edge_types']}")

    if 'time_span' in stats:
        print(f"\n时间跨度:")
        for key, value in stats['time_span'].items():
            print(f"  {key}: {value}")

    print(f"{'=' * 70}")


# 执行分析
print("\n" + "=" * 70)
print("COVID 数据集分析")
print("=" * 70)

# 分析原始数据
try:
    raw_stats = analyze_covid_raw_data()
    print_statistics(raw_stats)
except Exception as e:
    print(f"分析原始数据失败: {e}")

# 分析处理后的数据集
try:
    processed_stats = analyze_covid_processed_dataset()
    print_statistics(processed_stats)
except Exception as e:
    print(f"分析处理后数据失败: {e}")

# 分析统一格式数据集
try:
    uni_stats = analyze_covid_uni_dataset()
    print_statistics(uni_stats)
except Exception as e:
    print(f"分析统一格式数据失败: {e}")
