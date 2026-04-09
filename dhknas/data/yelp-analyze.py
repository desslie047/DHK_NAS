import os
import os.path as osp
import numpy as np
from collections import Counter
import torch
import json
from dhknas import config

dataroot = "./raw_data/"


def analyze_yelp_raw_data():
    """分析Yelp原始数据"""
    datafiles = {}
    for part in "business review tip user checkin".split():
        datafiles[part] = os.path.join(dataroot, f"yelp_academic_dataset_{part}.json")

    # 读取business数据
    businesses = []
    with open(datafiles["business"], "r") as f:
        for line in f:
            d = json.loads(line)
            businesses.append([d["business_id"], d["categories"]])

    # 读取review数据
    reviews = []
    with open(datafiles["review"], "r") as f:
        for line in f:
            d = json.loads(line)
            reviews.append([d["user_id"], d["business_id"], d["stars"], d["date"]])

    # 读取tip数据
    tips = []
    with open(datafiles["tip"], "r") as f:
        for line in f:
            d = json.loads(line)
            tips.append([d["user_id"], d["business_id"], d["date"]])

    print(f"原始数据统计:")
    print(f"  business: {len(businesses)}")
    print(f"  review: {len(reviews)}")
    print(f"  tip: {len(tips)}")

    return businesses, reviews, tips


def analyze_yelp_processed_dataset():
    """分析处理后的Yelp数据集"""

    # 模拟预处理流程
    businesses, reviews, tips = analyze_yelp_raw_data()

    # 选择的类别
    cates_included = [
        "American (New)",
        "Fast Food",
        "Sushi Bars",
    ]

    # 筛选business
    business_filtered = []
    for bid, cat in businesses:
        if cat:
            for cat_in in cates_included:
                if cat_in in cat:
                    business_filtered.append([bid, cat_in])
                    break

    bid_set = set(x[0] for x in business_filtered)
    category_counts = Counter([x[1] for x in business_filtered])

    # 筛选reviews (2012年数据)
    reviews_filtered = []
    for uid, bid, rate, date in reviews:
        y, m = date[:7].split("-")
        if int(y) == 2012 and bid in bid_set:
            reviews_filtered.append([uid, bid, int(m)])

    # 筛选tips (2012年数据)
    tips_filtered = []
    for uid, bid, date in tips:
        y, m = date[:7].split("-")
        if int(y) == 2012 and bid in bid_set:
            tips_filtered.append([uid, bid, int(m)])

    # 统计节点
    users = set([x[0] for x in reviews_filtered] + [x[0] for x in tips_filtered])
    items = bid_set

    # 统计时间
    times = set([x[2] for x in reviews_filtered] + [x[2] for x in tips_filtered])
    times = sorted(list(times))

    # 统计边
    review_edges = len(reviews_filtered)
    tip_edges = len(tips_filtered)
    interact_edges = review_edges + tip_edges

    stats = {
        'dataset_name': 'Yelp',
        'num_nodes': {
            'user': len(users),
            'item': len(items),
            'category': len(category_counts),
            'total': len(users) + len(items)
        },
        'num_edges': {
            'review (user-item)': review_edges,
            'tip (user-item)': tip_edges,
            'interact (user-item)': interact_edges,
            'total': interact_edges
        },
        'node_types': ['user', 'item'],
        'edge_types': ['review (user-item)', 'tip (user-item)', 'interact (user-item)'],
        'time_span': {
            'time_steps': len(times),
            'months': times,
            'range': f"{times[0]} - {times[-1]}",
            'duration': f"{len(times)} 个月",
            'year': 2012,
            'date_range': f"2012-{times[0]:02d} 到 2012-{times[-1]:02d}"
        },
        'categories': {
            'included': cates_included,
            'distribution': dict(category_counts)
        }
    }

    return stats


def analyze_yelp_loaded_dataset():
    """分析加载后的Yelp数据集"""
    from dhknas.data.yelp import YelpDataset

    dataset = YelpDataset(undirected=False, metapath2vec=True)
    data = dataset.dataset
    times = dataset.times()

    # 统计节点数量
    num_users = data['user'].num_nodes
    num_items = data['item'].num_nodes

    # 统计边数量
    num_review_edges = data['user', 'review', 'item'].edge_index.shape[1]
    num_tip_edges = data['user', 'tip', 'item'].edge_index.shape[1]
    num_interact_edges = data['user', 'interact', 'item'].edge_index.shape[1]

    # 统计类别
    categories = data['item'].y.unique()
    category_counts = Counter(data['item'].y.numpy())

    stats = {
        'dataset_name': 'Yelp (已加载)',
        'num_nodes': {
            'user': num_users,
            'item': num_items,
            'total': num_users + num_items
        },
        'num_edges': {
            'review (user-item)': num_review_edges,
            'tip (user-item)': num_tip_edges,
            'interact (user-item)': num_interact_edges,
            'total': num_interact_edges
        },
        'node_types': ['user', 'item'],
        'edge_types': ['review', 'tip', 'interact'],
        'time_span': {
            'time_steps': len(times),
            'times': times,
            'range': f"{times[0]} - {times[-1]}",
            'duration': f"{len(times)} 个时间步"
        },
        'item_features': {
            'num_categories': len(categories),
            'category_distribution': dict(category_counts)
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

    if 'categories' in stats:
        print(f"\n业务类别:")
        print(f"  包含类别: {stats['categories']['included']}")
        print(f"  类别分布: {stats['categories']['distribution']}")

    if 'item_features' in stats:
        print(f"\nItem特征:")
        print(f"  类别数量: {stats['item_features']['num_categories']}")
        print(f"  类别分布: {stats['item_features']['category_distribution']}")

    print(f"{'=' * 70}")


# 分析处理后的数据集
try:
    processed_stats = analyze_yelp_processed_dataset()
    print_statistics(processed_stats)
except Exception as e:
    print(f"分析处理后数据失败: {e}")

# 分析加载的数据集
try:
    loaded_stats = analyze_yelp_loaded_dataset()
    print_statistics(loaded_stats)
except Exception as e:
    print(f"分析加载数据失败: {e}")
