from matplotlib.pyplot import colormaps

from dhknas.models import DHSearcher
import torch
from copy import deepcopy
from dhknas.args_search import get_args
from dhknas.data import load_data
from dhknas.models.load_model import load_pre_post, load_lazy_hetero_weights
from dhknas.models.DHSpaceSearch import DHNet, DHSpace, DHSearcher
from dhknas.trainer import load_trainer
from dhknas.utils import setup_seed
import json
import os
from dhknas.models.KAA_GAT import KAAGATConv
from dhknas.models.DHSpace import DHSpaceKAA, DHNetKAA

# 2) 在文件内新增一个小工具函数（放在任意函数外部均可）
def _ensure_noise_defaults(args):
    if not hasattr(args, "noise_feat_std"):
        args.noise_feat_std = 0.0
    if not hasattr(args, "noise_edge_drop"):
        args.noise_edge_drop = 0.0
    if not hasattr(args, "noise_label_flip"):
        args.noise_label_flip = 0.0
    if not hasattr(args, "noise_apply_stage"):
        args.noise_apply_stage = "train"  # 可选: train|val|test|all
    if not hasattr(args, "robustness_seed"):
        args.robustness_seed = 3407


def evaluate_and_visualize(model, dataset, dataset_name):
    import matplotlib.pyplot as plt

    # 在函数开头添加中文字体设置
    import matplotlib.pyplot as plt
    import platform

    # 设置中文字体
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    with torch.no_grad():
        # 获取 HeteroData
        if hasattr(dataset, 'dataset'):
            data = dataset.dataset
            print(f"从 dataset.dataset 获取数据")
        else:
            data = dataset

        print(f"Data 类型: {type(data)}")

        # 分析异构动态图结构
        print("\n=== 异构动态图结构分析 ===")
        node_types = list(data.x_dict.keys()) if hasattr(data, 'x_dict') else []
        edge_types = list(data.edge_index_dict.keys()) if hasattr(data, 'edge_index_dict') else []

        print(f"节点类型: {node_types}")
        print(f"边类型: {edge_types}")

        # 检查时间信息 - 修改这里以避免触发 HeteroData 的自动收集
        has_time = False
        time_dict = {}

        # 方法1: 检查 _store 属性
        try:
            if hasattr(data, '_store') and hasattr(data._store, '__contains__'):
                if 'time_dict' in data._store:
                    has_time = True
                    time_dict = data._store['time_dict']
                    print(f"找到时间信息(time_dict): {list(time_dict.keys())}")
                elif 't_dict' in data._store:
                    has_time = True
                    time_dict = data._store['t_dict']
                    print(f"找到时间信息(t_dict): {list(time_dict.keys())}")
        except (AttributeError, KeyError):
            pass

        # 方法2: 直接检查各节点/边类型的时间属性
        if not has_time:
            for node_type in node_types:
                try:
                    node_store = data[node_type]
                    if hasattr(node_store, 't'):
                        has_time = True
                        time_dict[node_type] = node_store.t
                        print(f"找到节点 {node_type} 的时间属性: t")
                    elif hasattr(node_store, 'time'):
                        has_time = True
                        time_dict[node_type] = node_store.time
                        print(f"找到节点 {node_type} 的时间属性: time")
                except (AttributeError, KeyError):
                    continue

        if not has_time:
            print("警告: 数据中未找到时间信息,将以静态图方式可视化")

        for node_type in node_types:
            num_nodes = data.x_dict[node_type].size(0)
            print(f"  {node_type}: {num_nodes} 个节点")
            if node_type in time_dict:
                time_vals = time_dict[node_type]
                print(f"    时间范围: {time_vals.min().item():.2f} - {time_vals.max().item():.2f}")

        if not edge_types:
            print("错误: 无法找到图的边信息")
            return

        # 收集所有边、节点和时间信息
        all_edges = []
        node_type_map = {}  # 全局节点ID -> 节点类型
        node_time_map = {}  # 全局节点ID -> 时间戳
        global_node_id = 0
        node_id_mapping = {}  # (node_type, local_id) -> global_id

        # 为每种节点类型分配全局ID
        for node_type in node_types:
            num_nodes = data.x_dict[node_type].size(0)
            for local_id in range(num_nodes):
                node_id_mapping[(node_type, local_id)] = global_node_id
                node_type_map[global_node_id] = node_type

                # 记录时间信息
                if node_type in time_dict:
                    node_time_map[global_node_id] = time_dict[node_type][local_id].item()
                else:
                    node_time_map[global_node_id] = 0.0

                global_node_id += 1

        print(f"\n总节点数: {global_node_id}")

        # 转换所有边到全局ID(包含边的时间信息)
        edge_time_dict = {}

        # 检查边的时间信息
        for edge_type in edge_types:
            try:
                edge_store = data[edge_type]
                if hasattr(edge_store, 't'):
                    edge_time_dict[edge_type] = edge_store.t
                elif hasattr(edge_store, 'time'):
                    edge_time_dict[edge_type] = edge_store.time
            except (AttributeError, KeyError):
                continue

        for edge_type in edge_types:
            src_type, relation, dst_type = edge_type
            edge_index = data.edge_index_dict[edge_type]
            edge_times = edge_time_dict.get(edge_type, None)

            print(f"\n边类型 {edge_type}: {edge_index.shape[1]} 条边")

            for i in range(edge_index.shape[1]):
                src_local = edge_index[0, i].item()
                dst_local = edge_index[1, i].item()

                src_global = node_id_mapping[(src_type, src_local)]
                dst_global = node_id_mapping[(dst_type, dst_local)]

                edge_time = edge_times[i].item() if edge_times is not None else 0.0
                all_edges.append((src_global, dst_global, edge_type, edge_time))

        print(f"\n总边数: {len(all_edges)}")

        # 智能采样策略(考虑时间维度)
        max_nodes = 80
        if global_node_id > max_nodes:
            print(f"\n节点数 {global_node_id} 过大,采样到 {max_nodes} 个节点...")

            from collections import defaultdict
            import numpy as np

            # 按时间分桶采样
            if has_time and node_time_map:
                # 将节点按时间分成时间片
                time_values = list(node_time_map.values())
                time_min, time_max = min(time_values), max(time_values)
                n_time_slices = 5  # 分成5个时间片

                time_slice_nodes = defaultdict(list)
                for node_id, time_val in node_time_map.items():
                    if time_max > time_min:
                        slice_idx = int((time_val - time_min) / (time_max - time_min + 1e-6) * n_time_slices)
                        slice_idx = min(slice_idx, n_time_slices - 1)
                    else:
                        slice_idx = 0
                    time_slice_nodes[slice_idx].append(node_id)

                # 从每个时间片采样
                sampled_nodes = set()
                nodes_per_slice = max_nodes // n_time_slices

                for slice_idx in sorted(time_slice_nodes.keys()):
                    slice_nodes = time_slice_nodes[slice_idx]
                    # 按度数排序选择重要节点
                    node_degrees = defaultdict(int)
                    for src, dst, _, _ in all_edges:
                        if src in slice_nodes:
                            node_degrees[src] += 1
                        if dst in slice_nodes:
                            node_degrees[dst] += 1

                    top_nodes = sorted(slice_nodes, key=lambda x: node_degrees[x], reverse=True)[:nodes_per_slice]
                    sampled_nodes.update(top_nodes)
                    print(f"  时间片 {slice_idx}: 采样 {len(top_nodes)} 个节点")

            else:
                # 退回到按度数采样
                node_degrees = defaultdict(int)
                for src, dst, _, _ in all_edges:
                    node_degrees[src] += 1
                    node_degrees[dst] += 1

                nodes_by_type = defaultdict(list)
                for node_id, node_type in node_type_map.items():
                    nodes_by_type[node_type].append((node_id, node_degrees[node_id]))

                sampled_nodes = set()
                quota_per_type = max_nodes // len(node_types)

                for node_type in node_types:
                    nodes = nodes_by_type[node_type]
                    nodes_sorted = sorted(nodes, key=lambda x: x[1], reverse=True)
                    for node_id, degree in nodes_sorted[:quota_per_type]:
                        sampled_nodes.add(node_id)

            print(f"采样节点数: {len(sampled_nodes)}")
            for node_type in node_types:
                type_count = sum(1 for nid in sampled_nodes if node_type_map[nid] == node_type)
                print(f"  {node_type}: {type_count} 个节点")

            # 过滤边
            sampled_edges = [(src, dst, etype, etime) for src, dst, etype, etime in all_edges
                             if src in sampled_nodes and dst in sampled_nodes]

            print(f"采样后边数: {len(sampled_edges)}")

            # 重新映射节点ID
            old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(sampled_nodes))}
            remapped_edges = [(old_to_new[src], old_to_new[dst], etype, etime)
                              for src, dst, etype, etime in sampled_edges]
            remapped_node_type_map = {old_to_new[old_id]: node_type_map[old_id]
                                      for old_id in sampled_nodes}
            remapped_node_time_map = {old_to_new[old_id]: node_time_map[old_id]
                                      for old_id in sampled_nodes}
        else:
            remapped_edges = [(src, dst, etype, etime) for src, dst, etype, etime in all_edges]
            remapped_node_type_map = node_type_map
            remapped_node_time_map = node_time_map

        if not remapped_edges:
            print("警告: 采样后没有边!")
            return

        # 创建NetworkX图
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import numpy as np

        G = nx.MultiDiGraph()

        # 添加节点(带类型和时间属性)
        for node_id, node_type in remapped_node_type_map.items():
            G.add_node(node_id, node_type=node_type, time=remapped_node_time_map[node_id])

        # 添加边(带类型和时间属性)
        for src, dst, edge_type, edge_time in remapped_edges:
            G.add_edge(src, dst, edge_type=str(edge_type), time=edge_time)

        print(f"\nNetworkX图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

        # 创建保存目录
        os.makedirs('results', exist_ok=True)
        save_path = f'results/graph_{dataset_name}_hetero_temporal.png'

        # 设置节点类型颜色映射
        unique_node_types = sorted(set(remapped_node_type_map.values()))
        node_type_colors = colormaps['Set3'](np.linspace(0, 1, len(unique_node_types)))
        node_type_to_color = {nt: node_type_colors[i] for i, nt in enumerate(unique_node_types)}

        # 创建时间颜色映射(用于节点边框)
        if has_time and remapped_node_time_map:
            time_values = list(remapped_node_time_map.values())
            time_norm = Normalize(vmin=min(time_values), vmax=max(time_values))
            time_cmap = colormaps['viridis']
        else:
            time_norm = None
            time_cmap = None

        # 创建图形(使用子图布局)
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[20, 1],
                              hspace=0.3, wspace=0.05)

        ax_main = fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[1, 0])
        ax_colorbar = fig.add_subplot(gs[0, 1])

        # 使用kamada_kawai布局
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # 根据时间给边着色
        edge_times = [data.get('time', 0) for _, _, data in G.edges(data=True)]
        if edge_times and max(edge_times) > min(edge_times):
            edge_time_norm = Normalize(vmin=min(edge_times), vmax=max(edge_times))
            edge_colors = [time_cmap(edge_time_norm(t)) for t in edge_times]
        else:
            edge_colors = ['lightgray'] * len(edge_times)

        # 绘制边
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=1.5,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax_main
        )

        # 绘制节点
        node_colors = [node_type_to_color[remapped_node_type_map[node]] for node in G.nodes()]

        # 节点边框颜色表示时间
        if time_cmap and time_norm:
            edge_colors_nodes = [time_cmap(time_norm(remapped_node_time_map[node])) for node in G.nodes()]
        else:
            edge_colors_nodes = 'black'

        node_sizes = [200 + G.degree(node) * 30 for node in G.nodes()]

        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors=edge_colors_nodes,
            linewidths=3.0,
            alpha=0.9,
            ax=ax_main
        )

        # 添加节点标签
        if len(G.nodes()) <= 100:
            labels = {}
            for node in G.nodes():
                node_type = remapped_node_type_map[node]
                node_time = remapped_node_time_map[node]
                if has_time:
                    labels[node] = f"{node}\n{node_type[:3]}\nt={node_time:.1f}"
                else:
                    labels[node] = f"{node}\n{node_type[:3]}"

            nx.draw_networkx_labels(G, pos, labels, font_size=6,
                                    font_color='black', ax=ax_main)

        # 添加时间颜色条
        if time_cmap and time_norm:
            sm = ScalarMappable(cmap=time_cmap, norm=time_norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=ax_colorbar)
            cbar.set_label('时间戳', fontsize=12)

        # 创建详细图例
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        ax_legend.axis('off')

        legend_elements = []
        for nt in unique_node_types:
            legend_elements.append(Patch(facecolor=node_type_to_color[nt],
                                         edgecolor='black', linewidth=2,
                                         label=f'节点类型: {nt}'))

        unique_edge_types = set(str(etype) for _, _, etype, _ in remapped_edges)
        for i, etype in enumerate(sorted(unique_edge_types)):
            relation = etype.split("'")[3] if "'" in etype else etype
            legend_elements.append(Line2D([0], [0], color='gray', linewidth=2,
                                          linestyle='-', label=f'边类型: {relation}'))

        legend_elements.append(Line2D([0], [0], color='none',
                                      label='\n节点边框颜色 = 时间戳'))
        legend_elements.append(Line2D([0], [0], color='none',
                                      label='节点大小 = 度数'))

        ax_legend.legend(handles=legend_elements, loc='center',
                         fontsize=10, ncol=3, frameon=True)

        # 统计信息
        from collections import Counter
        node_type_counts = Counter(remapped_node_type_map.values())
        edge_type_counts = Counter(str(etype) for _, _, etype, _ in remapped_edges)

        stats_lines = []
        stats_lines.append("节点统计: " + ", ".join([f"{nt}: {cnt}" for nt, cnt in node_type_counts.items()]))

        edge_stats = []
        for et, cnt in list(edge_type_counts.items())[:3]:
            relation = et.split("'")[3] if "'" in et else et
            edge_stats.append(f"{relation}: {cnt}")
        stats_lines.append("边统计: " + ", ".join(edge_stats))

        if has_time:
            time_values = list(remapped_node_time_map.values())
            stats_lines.append(f"时间范围: {min(time_values):.2f} - {max(time_values):.2f}")

        title = f'{dataset_name} 异构动态图结构可视化\n' + '\n'.join(stats_lines)
        ax_main.set_title(title, fontsize=16, pad=20)
        ax_main.axis('off')

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f'\n可视化已保存到 {save_path}')

        # 生成时间序列分析图
        if has_time and len(remapped_edges) > 0:
            generate_temporal_analysis(remapped_edges, remapped_node_type_map,
                                       dataset_name, save_dir='results')

        # 自动打开图像
        try:
            import platform
            if platform.system() == 'Windows':
                os.startfile(os.path.abspath(save_path))
            else:
                import subprocess
                subprocess.run(['xdg-open', os.path.abspath(save_path)])
        except Exception as e:
            print(f'无法自动打开图片: {e}')


def generate_temporal_analysis(edges, node_type_map, dataset_name, save_dir='results'):
    """生成时间序列分析图"""
    print(f'生成 {dataset_name} 的时间序列分析图...')
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    import platform

    # 设置中文字体
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False


    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name} 时间序列分析', fontsize=16)

    # 提取时间信息
    edge_times = [etime for _, _, _, etime in edges]
    edge_types = [str(etype) for _, _, etype, _ in edges]

    # 1. 边创建时间分布
    ax = axes[0, 0]
    ax.hist(edge_times, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('时间戳', fontsize=12)
    ax.set_ylabel('边数量', fontsize=12)
    ax.set_title('边创建时间分布', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 2. 不同边类型的时间分布
    ax = axes[0, 1]
    unique_edge_types = list(set(edge_types))
    for etype in unique_edge_types[:5]:  # 只显示前5种
        etype_times = [etime for _, _, et, etime in edges if str(et) == etype]
        relation = etype.split("'")[3] if "'" in etype else etype
        ax.hist(etype_times, bins=30, alpha=0.5, label=relation[:20])
    ax.set_xlabel('时间戳', fontsize=12)
    ax.set_ylabel('频率', fontsize=12)
    ax.set_title('不同边类型的时间分布', fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. 累积边数量随时间变化
    ax = axes[1, 0]
    sorted_times = sorted(edge_times)
    cumulative_edges = np.arange(1, len(sorted_times) + 1)
    ax.plot(sorted_times, cumulative_edges, linewidth=2, color='darkgreen')
    ax.set_xlabel('时间戳', fontsize=12)
    ax.set_ylabel('累积边数量', fontsize=12)
    ax.set_title('图增长趋势', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 4. 时间窗口内的活跃度
    ax = axes[1, 1]
    if len(edge_times) > 10:
        time_windows = np.linspace(min(edge_times), max(edge_times), 20)
        activity = []
        for i in range(len(time_windows) - 1):
            count = sum(1 for t in edge_times if time_windows[i] <= t < time_windows[i + 1])
            activity.append(count)

        ax.bar(range(len(activity)), activity, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('时间窗口', fontsize=12)
        ax.set_ylabel('边数量', fontsize=12)
        ax.set_title('时间窗口活跃度', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = f'{save_dir}/temporal_analysis_{dataset_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'时间序列分析已保存到 {save_path}')



if __name__ == "__main__":
    args = get_args()
    args.use_kaa = getattr(args, 'use_kaa', False)
    args.kan_layers = getattr(args, 'kan_layers', 2)
    args.grid_size = getattr(args, 'grid_size', 1)
    args.spline_order = getattr(args, 'spline_order', 1)

    # dataset
    dataset, args = load_data(args)
    hid_dim, metadata, twin, KTO, KN, KR, n_heads, predict_type, device = (
        args.hid_dim,
        dataset.metadata,
        args.twin,
        args.KTO,
        args.KN,
        args.KR,
        args.n_heads,
        args.predict_type,
        args.device,
    )

    # model
    setup_seed(args.seed)
    featemb, nclf_linear = load_pre_post(args, dataset)
    n_layers = 2  # reuse the searched first layer for efficiency

    # 在这里添加条件判断使用DHNetKAA还是DHNet
    if args.use_kaa:
        print('使用DHNetKAA模型')
        test_space = DHSpaceKAA(
            hid_dim, metadata, twin, KTO, KN, KR, n_heads,
            causal_mask=args.causal_mask, last_mask=False, full_mask=True,
            rel_time_type=args.rel_time_type, time_patch_num=args.patch_num,
            norm=args.norm, hupdate=args.hupdate, kan_layers=args.kan_layers,
            grid_size=args.grid_size, spline_order=args.spline_order
        )
        # 验证alpha参数是否存在
        alpha_params = [n for n, p in test_space.named_parameters() if "alpha" in n]
        print("找到的alpha参数:", alpha_params)
        if not alpha_params:
            raise ValueError("DHSpaceKAA类没有任何alpha参数，优化器初始化将失败")
        Net = DHNetKAA
        dhspaces = []
        for i in range(n_layers - 1):
            dhspaces.append(
                DHSpaceKAA(
                    hid_dim,
                    metadata,
                    twin,
                    KTO,
                    KN,
                    KR,
                    n_heads,
                    causal_mask=args.causal_mask,
                    last_mask=False,
                    full_mask=True,
                    rel_time_type=args.rel_time_type,
                    time_patch_num=args.patch_num,
                    norm=args.norm,
                    hupdate=args.hupdate,
                    kan_layers=args.kan_layers,
                    grid_size=args.grid_size,
                    spline_order=args.spline_order,
                )
            )
        dhspaces.append(
            DHSpaceKAA(
                hid_dim,
                metadata,
                twin,
                KTO,
                KN,
                KR,
                n_heads,
                causal_mask=args.causal_mask,
                last_mask=True,
                full_mask=True,
                rel_time_type="relative",
                time_patch_num=1,
                norm=args.norm,
                hupdate=args.hupdate,
                kan_layers=args.kan_layers,
                grid_size=args.grid_size,
                spline_order=args.spline_order,
            )
        )
    else:
        print('使用DHSpace原版模型')
        Net = DHNet
        dhspaces = []
        for i in range(n_layers - 1):
            dhspaces.append(
                DHSpace(
                    hid_dim,
                    metadata,
                    twin,
                    KTO,
                    KN,
                    KR,
                    n_heads,
                    causal_mask=args.causal_mask,
                    last_mask=False,
                    full_mask=True,
                    rel_time_type=args.rel_time_type,
                    time_patch_num=args.patch_num,
                    norm=args.norm,
                    hupdate=args.hupdate,
                )
            )
        dhspaces.append(
            DHSpace(
                hid_dim,
                metadata,
                twin,
                KTO,
                KN,
                KR,
                n_heads,
                causal_mask=args.causal_mask,
                last_mask=True,
                full_mask=True,
                rel_time_type="relative",
                time_patch_num=1,
                norm=args.norm,
                hupdate=args.hupdate,
            )
        )

    # 修改模型初始化，添加KAA相关参数
    if args.use_kaa:
        model = Net(
            hid_dim,
            twin,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
            kan_layers=args.kan_layers,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
        )
    else:
        model = Net(
            hid_dim,
            twin,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )

    # device
    model = model.to(device)
    dataset.to(device)
    if args.resume:
        model.load_state_dict(
            torch.load(os.path.join(args.supernet_dir, f"checkpoint{args.resume}"))
        )

    # 3) 在 dataset.to(device) 之后，searcher.search(...) 之前，加入以下代码
    # _ensure_noise_defaults(args)
    # for _stage in ("train", "val", "test"):
    #     apply_noise(dataset, args, stage=_stage)

    # trainer
    if args.dataset == "Covid":
        # Set regression task for COVID dataset
        args.task = "regression"
        args.out_dim = 1  # Ensure output dimension is 1 for regression

        # Use MSE loss for regression
        from torch import nn

        criterion = nn.MSELoss()
        trainer, _ = load_trainer(args)  # Ignore the criterion from load_trainer
    else:
        # Use default trainer and criterion for other datasets
        trainer, criterion = load_trainer(args)

    searcher = DHSearcher(
        criterion, args.supernet_dir, None, n_warmup=args.n_warmup, args=args
    )

    best_pop = searcher.search(model, dhspaces, dataset, topk=args.topk)

    if args.visualize:
        print("=" * 80)
        print("调试信息:")
        print(f"Dataset 类型: {type(dataset)}")
        if hasattr(dataset, 'data'):
            print(f"Dataset.data 类型: {type(dataset.data)}")
            print(f"Dataset.data 属性: {[a for a in dir(dataset.data) if not a.startswith('_')][:20]}")

        # 尝试访问第一个数据样本
        try:
            sample = dataset[0] if hasattr(dataset, '__getitem__') else dataset
            print(f"Sample 类型: {type(sample)}")
            print(f"Sample 有 edge_index_dict: {hasattr(sample, 'edge_index_dict')}")
            print(f"Sample 有 edge_index: {hasattr(sample, 'edge_index')}")
        except Exception as e:
            print(f"访问样本失败: {e}")

        print("=" * 80)

        # 然后再调用可视化
        evaluate_and_visualize(model, dataset, f"{args.dataset}_searched")

    # logs
    info_dict = args.__dict__
    configs = []
    for i, (config, estimation) in enumerate(best_pop):
        arch_info_dict = deepcopy(info_dict)
        arch_dir = os.path.join(args.arch_dir, f"{i}")
        os.makedirs(arch_dir, exist_ok=True)
        fconfig = os.path.join(arch_dir, "config")
        torch.save(config, fconfig)
        json.dump(
            info_dict,
            open(os.path.join(arch_dir, "supernet.json"), "w"),
            indent=4,
            sort_keys=True,
        )
        open(os.path.join(arch_dir, "config_read.txt"), "w").write(f"{config}")
        configs.append(arch_dir)

    # rerun the searched model
    torch.cuda.empty_cache()
    dev = args.device.split(":")[-1]
    arch_dir = configs[0]
    results = []


    # [11, 22, 33]
    for seed in [11, 22, 33, 0]:
        noise_flags = (
            f" --noise_feat_std {args.noise_feat_std}"
            f" --noise_edge_drop {args.noise_edge_drop}"
            f" --noise_label_flip {args.noise_label_flip}"
            f" --noise_apply_stage {args.noise_apply_stage}"
            f" --robustness_seed {args.robustness_seed}"
        )

        log_dir = f"{arch_dir}/{seed}"
        os.makedirs(log_dir, exist_ok=True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels
        run_model_path = os.path.join(root_dir, "scripts", "run", "run_model.py")
        # cmd = f'python "{run_model_path}" --seed {seed} --device {dev} --model DHSpace --twin {args.twin} --log_dir "{log_dir}" --dhconfig "{arch_dir}" --dataset {args.dataset} --n_heads {args.n_heads} --norm {args.norm} --hlinear_act {args.hlinear_act} --lr {args.lr} --wd {args.wd} > "{log_dir}/log.txt"'
        # if args.use_kaa:
        #     # 将KAA相关参数保存到配置文件，而不是作为命令行参数传递
        #     # 在dhconfig目录下创建kaa_config.json
        #     kaa_config = {
        #         "use_kaa": True,
        #         "kan_layers": args.kan_layers,
        #         "grid_size": args.grid_size,
        #         "spline_order": args.spline_order
        #     }
        #     with open(os.path.join(arch_dir, "kaa_config.json"), "w") as f:
        #         json.dump(kaa_config, f)
        #
        #     # 不传递KAA参数，避免unrecognized arguments错误
        #     cmd = f'python "{run_model_path}" --seed {seed} --device {dev} --model DHSpace --twin {args.twin} --log_dir "{log_dir}" --dhconfig "{arch_dir}" --dataset Covid --n_heads {args.n_heads} --norm {args.norm} --hlinear_act {args.hlinear_act} --lr {args.lr} --wd {args.wd} --task regression > "{log_dir}/log.txt"'
        # else:
        #     cmd = f'python "{run_model_path}" --seed {seed} --device {dev} --model DHSpace --twin {args.twin} --log_dir "{log_dir}" --dhconfig "{arch_dir}" --dataset Covid --n_heads {args.n_heads} --norm {args.norm} --hlinear_act {args.hlinear_act} --lr {args.lr} --wd {args.wd} --task regression > "{log_dir}/log.txt"'

        if args.dataset == "Covid":
            # Add regression task parameter for COVID dataset
            if args.use_kaa:
                # Create a KAA config file in the arch directory
                kaa_config = {
                    "use_kaa": True,
                    "kan_layers": args.kan_layers,
                    "grid_size": args.grid_size,
                    "spline_order": args.spline_order
                }
                with open(os.path.join(arch_dir, "kaa_config.json"), "w") as f:
                    json.dump(kaa_config, f)
                cmd = (f'python "{run_model_path}" --seed {seed} --device {dev} --model DHSpaceKAA --twin {args.twin} '
                       f'--log_dir "{log_dir}" --dhconfig "{arch_dir}" --dataset Covid --n_heads {args.n_heads} '
                       f'--norm {args.norm} --hlinear_act {args.hlinear_act} --lr {args.lr} --wd {args.wd} ')
                cmd += noise_flags
            else:
                cmd = (f'python "{run_model_path}" --seed {seed} --device {dev} --model DHSpace --twin {args.twin} '
                       f'--log_dir "{log_dir}" --dhconfig "{arch_dir}" --dataset Covid --n_heads {args.n_heads} '
                       f'--norm {args.norm} --hlinear_act {args.hlinear_act} --lr {args.lr} --wd {args.wd} ')
                cmd += noise_flags

        else:
            # Use regular command without regression task for other datasets
            if args.use_kaa:
                # Create a KAA config file in the arch directory
                kaa_config = {
                    "use_kaa": True,
                    "kan_layers": args.kan_layers,
                    "grid_size": args.grid_size,
                    "spline_order": args.spline_order
                }
                with open(os.path.join(arch_dir, "kaa_config.json"), "w") as f:
                    json.dump(kaa_config, f)
                cmd = (f'python "{run_model_path}" --seed {seed} --device {dev} --model DHSpaceKAA --twin {args.twin} '
                       f'--log_dir "{log_dir}" --dhconfig "{arch_dir}" --dataset {args.dataset} --n_heads {args.n_heads} '
                       f'--norm {args.norm} --hlinear_act {args.hlinear_act} --lr {args.lr} --wd {args.wd}')
                cmd += noise_flags
            else:
                cmd = (f'python "{run_model_path}" --seed {seed} --device {dev} --model DHSpace --twin {args.twin} '
                       f'--log_dir "{log_dir}" --dhconfig "{arch_dir}" --dataset {args.dataset} --n_heads {args.n_heads} '
                       f'--norm {args.norm} --hlinear_act {args.hlinear_act} --lr {args.lr} --wd {args.wd}')
                cmd += noise_flags

        with open(os.path.join(log_dir, "cmd.sh"), "w") as f:
            f.write(cmd)
        os.system(cmd)

        try:
            with open(os.path.join(log_dir, "info.json"), 'r') as f:
                info = json.load(f)
                if args.dataset == "Covid":
                    # For regression tasks (COVID), use MSE or MAE
                    results.append(info.get("test_mse", info.get("test_mae", info.get("test_rmse", None))))
                else:
                    # For classification tasks (other datasets), use AUC
                    results.append(info.get("test_auc", None))
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Couldn't load results for seed {seed}: {e}")
            results.append(None)

        # results.append(json.load(open(os.path.join(log_dir, "info.json")))["test_auc"])
    with open(os.path.join(arch_dir, "results.txt"), "w") as f:
        f.write(str(results))
