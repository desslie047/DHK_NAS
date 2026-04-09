import argparse
from dhknas.models import Sta_MODEL, Homo_MODEL
import os


def setargs(args, hp):
    for k, v in hp.items():
        setattr(args, k, v)


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cfg", type=int, default=1)
    # basic
    parser.add_argument("--dataset", type=str, default="Aminer")
    parser.add_argument("--model", type=str, default="DHSpace")
    parser.add_argument("--dhconfig", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs/tmp")
    parser.add_argument("--device", default="6")
    parser.add_argument("--seed", type=int, default=22)

    # auto
    parser.add_argument("--dynamic", type=int, default=-1)
    parser.add_argument("--homo", type=int, default=-1)
    parser.add_argument("--twin", type=int, default=-1)
    parser.add_argument("--test_full", type=int, default=-1)
    parser.add_argument("--predict_type", type=str, default="")
    parser.add_argument("--in_dim", type=int, default=-1)
    parser.add_argument("--hid_dim", type=int, default=-1)
    parser.add_argument("--out_dim", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=-1)

    # optim
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--cul", type=int, default=1)

    # hp
    parser.add_argument("--norm", type=int, default=1)
    parser.add_argument("--hlinear_act", type=str, default="tanh")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--grad_clip", type=float, default=0)

    # search
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--supernet_early_stop", type=int, default=1000)
    parser.add_argument("--causal_mask", type=int, default=1, help="1 True else False")
    parser.add_argument("--node_entangle_type", type=str, default="None")
    parser.add_argument("--rel_entangle_type", type=str, default="None")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--rel_time_type", type=str, default="relative")
    parser.add_argument("--hupdate", type=int, default=1)
    parser.add_argument("--reset_type", type=int, default=0)
    parser.add_argument("--reset_type2", type=int, default=0)
    parser.add_argument("--patch_num", type=int, default=1)
    parser.add_argument("--KN", type=int, default=2)
    parser.add_argument("--KR", type=int, default=2)
    parser.add_argument("--KTO", type=int, default=10)
    parser.add_argument("--n_warmup", type=int, default=40)
    parser.add_argument("--arch_dir", type=str, default="")
    parser.add_argument("--supernet_dir", type=str, default="")

    # KAA相关参数
    parser.add_argument('--use_kaa', action='store_true', help='Use KAA attention mechanism')
    parser.add_argument('--kan_layers', type=int, default=2, help='Number of KAN layers')
    parser.add_argument('--grid_size', type=int, default=1, help='Grid size for KAN')
    parser.add_argument('--spline_order', type=int, default=1, help='Spline order for KAN')

    # 消融实验参数
    parser.add_argument('--ablation_no_kaa', action='store_true', help='Disable KAA mechanism for ablation study')
    parser.add_argument('--ablation_no_attention_loc', action='store_true',
                        help='Disable attention localization for ablation study')
    parser.add_argument('--ablation_no_hetero', action='store_true',
                        help='Disable heterogeneous information for ablation study')


    args = parser.parse_args(args)

    # full cfg
    if args.use_cfg:
        if args.dataset == "Aminer":
            hp = {
                "patch_num": 3,
                "KN": 5,
                "KR": 4,
                "KTO": 400,
                "n_layers": 2,
                "n_heads": 4,
                "n_warmup": 30,
                "twin": 8,
            }
            # hp = {
            #     "patch_num": 2,
            #     "KN": 5,
            #     "KR": 4,
            #     "KTO": 500,
            #     "n_layers": 3,
            #     "n_heads": 4,
            #     "n_warmup": 30,
            #     "twin": 8,
            # }
        elif args.dataset == "Ecomm":
            hp = {
                "patch_num": 2,
                "KN": 5,
                "KR": 3,
                "KTO": 500,
                "n_layers": 2,
                "n_heads": 2,
                "n_warmup": 15,
                "twin": 7,
            }
        elif args.dataset == "Yelp-nc":
            hp = {
                "patch_num": 2,
                "KN": 5,
                "KR": 5,
                "KTO": 500,
                "n_layers": 2,
                "n_heads": 2,
                "n_warmup": 20,
                "twin": 12,
            }
        elif args.dataset == "Covid":
            hp = {
                "patch_num": 2,
                "KN": 4,
                "KR": 4,
                "KTO": 500,
                "n_layers": 3,
                "n_heads": 8,
                "n_warmup": 30,
                "twin": 5,
                "lr": 0.001,
                "wd": 0.0005,
                "patience": 30,
                "hid_dim": 32,
                "norm": 1,  # 使用批归一化
            }
        else:
            raise NotImplementedError(f"dataset {args.dataset} not implemented")
        setargs(args, hp)

    # post
    # 消融实验参数处理
    if args.ablation_no_kaa:
        args.use_kaa = False

    if args.ablation_no_attention_loc:
        # 完全禁用注意力定位机制
        args.causal_mask = 0  # 禁用因果掩码
        args.topk = -1  # 使用全部节点而不是topk选择
        # 禁用时间相关的定位编码
        # args.rel_time_type = "independent"  # 禁用相对时间编码
        # 禁用空间定位机制
        args.patch_num = 1  # 单个patch，避免空间分割定位
        # 禁用层次化的注意力更新机制
        args.hupdate = 0  # 禁用层次更新
        # 禁用重置机制（可能影响注意力的定位性）
        # args.reset_type = 0
        # args.reset_type2 = 0
        # 简化为单头注意力，避免多头带来的局部化
        args.n_heads = 1
        # 如果有其他位置编码或定位相关参数，也应该在这里禁用

    if args.ablation_no_hetero:
        # 禁用异质信息，将entangle类型设为None
        args.node_entangle_type = "None"
        args.rel_entangle_type = "None"
        # 禁用参数空间相关的异质信息
        args.KN = 1  # 将节点相关的K设为1，退化为同质
        args.KR = 1  # 将关系相关的K设为1，退化为同质
        args.patch_num = 1  # 将patch数量设为1，简化异质处理
        # 如果有其他异质信息相关的参数，也应该在这里设置
        args.n_heads = 1  # 单头注意力，避免多头带来的异质性

    assert args.model == "DHSpace", "DHSearcher only supports DHSpace"
    args.device = f"cuda:{args.device}"
    args.dynamic = args.model not in Sta_MODEL
    args.homo = args.model in Homo_MODEL
    args.test_full = not args.dynamic  # static model use full training data for testing
    os.makedirs(args.log_dir, exist_ok=True)
    args.supernet_dir = os.path.join(args.log_dir, "supernet/")
    args.arch_dir = os.path.join(args.log_dir, "archs/")

    for d in [args.log_dir, args.supernet_dir, args.arch_dir]:
        os.makedirs(d, exist_ok=True)


    return args
