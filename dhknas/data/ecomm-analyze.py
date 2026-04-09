import os.path as osp
import numpy as np
from collections import Counter
from dhknas import config


def analyze_ecomm_dataset():
    """分析Ecomm数据集的特征"""

    dataroot = osp.join(config.dataroot, "ecomm")
    datafile = osp.join(dataroot, "ecomm_edge_train.txt")

    # 读取原始数据
    data = []
    with open(datafile, "r") as file:
        for line in file:
            nid1, nid2, etype, t = line.split()
            data.append([nid1, nid2, etype, int(t)])
    data = np.array(data)

    # 统计
    cnt = {}
    for i in range(4):
        cnt[i] = Counter(data[:, i])

    # 节点统计
    node1_num = len(cnt[0])  # user数量
    node2_num = len(cnt[1])  # item数量

    # 边类型统计 - 动态处理所有边类型
    edge_type_names = {
        '0': 'click',
        '1': 'buy',
        '2': 'cart',
        '3': 'favorite',
        '4': 'unknown_type_4'  # 添加未知类型
    }

    # 处理所有边类型，包括未定义的
    edge_counts = {}
    for k, v in cnt[2].items():
        edge_name = edge_type_names.get(k, f'type_{k}')  # 如果类型未定义，使用 type_X
        edge_counts[edge_name] = v

    # 时间跨度统计
    time_stamps = sorted([int(t) for t in cnt[3].keys()])

    # 生成边类型列表
    edge_type_list = [f"{edge_counts.keys()} (user-item)"]

    stats = {
        'dataset_name': 'Ecomm',
        'num_nodes': {
            'user': node1_num,
            'item': node2_num,
            'total': node1_num + node2_num
        },
        'num_edges': {
            **edge_counts,
            'total': len(data)
        },
        'node_types': ['user', 'item'],
        'edge_types': [f"{name} (user-item)" for name in edge_counts.keys()],
        'time_span': {
            'time_steps': len(time_stamps),
            'timestamps': time_stamps,
            'range': f"{time_stamps[0]} - {time_stamps[-1]}",
            'duration': f"{len(time_stamps)} 天",
            'date_range': f"2019-06-{time_stamps[0] % 100:02d} 到 2019-06-{time_stamps[-1] % 100:02d}"
        }
    }

    return stats


def print_statistics(stats):
    """打印统计结果"""
    print(f"\n{'=' * 70}")
    print(f"数据集: {stats['dataset_name']}")
    print(f"\n节点数量:")
    for node_type, count in stats['num_nodes'].items():
        print(f"  {node_type}: {count}")

    print(f"\n边数量:")
    for edge_type, count in stats['num_edges'].items():
        print(f"  {edge_type}: {count}")

    print(f"\n节点类型: {stats['node_types']}")
    print(f"\n边类型:")
    for et in stats['edge_types']:
        print(f"  - {et}")

    print(f"\n时间跨度:")
    for key, value in stats['time_span'].items():
        print(f"  {key}: {value}")
    print(f"{'=' * 70}")


# 执行分析
ecomm_stats = analyze_ecomm_dataset()
print_statistics(ecomm_stats)
