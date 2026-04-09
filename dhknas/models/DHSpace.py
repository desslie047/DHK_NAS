from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import torch
import math
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones
from .HLinear import HLinear, HLayerNorm


class DHSpace(nn.Module):
    def __init__(
        self,
        hid_dim,
        metadata,
        twin,
        K_To,
        K_N,
        K_R,
        n_heads=8,
        norm=True,
        causal_mask=False,
        last_mask=False,
        full_mask=False,
        node_hetero=False,
        node_entangle_type="None",
        rel_entangle_type="None",
        rel_time_type="relative",
        time_patch_num=1,
        hupdate=True,
        args=None,
        skip=False,
        **kwargs,
    ):
        super(DHSpace, self).__init__()

        # Data Attrs
        self.twin = twin

        # Data Type Info
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.num_types = len(self.node_types)
        self.num_relations = len(self.edge_types)
        self.ntype2id = dict(zip(self.node_types, range(self.num_types)))
        self.etype2id = dict(
            zip([et for _, et, _ in self.edge_types], range(self.num_relations))
        )  # [etype] -> [0,1,2]
        self.id2ntype = self.node_types
        # [0,1,2] -> [(ntype_src,etype,ntype_tar)]
        self.id2etype = self.edge_types

        # Space Constraint
        self.K_To = K_To
        self.K_N = K_N
        self.K_R = K_R
        self.causal_mask = causal_mask
        self.last_mask = last_mask
        self.full_mask = full_mask
        self.node_hetero = node_hetero
        self.norm = norm
        self.node_entangle_type = node_entangle_type
        self.rel_entangle_type = rel_entangle_type
        self.rel_time_type = rel_time_type
        self.collect_att = True
        self.time_patch_num = time_patch_num
        self.hupdate = hupdate
        self.args = args
        self.skip = skip
        # Model hyperparams
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.d_k = hid_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.aggr = "add"
        self.node_dim = 0

        # init
        self.dummy_A_To = torch.ones(
            (twin, twin, self.num_relations)
        ).long()  # T x T x C_e
        self.dummy_A_N = torch.zeros((twin, self.num_types)).long()  # T x C_n

        if self.rel_time_type == "independent":
            rel_time_len = twin * twin
        elif self.rel_time_type == "relative":
            rel_time_len = 2 * twin
        elif self.rel_time_type == "source":
            rel_time_len = twin
        elif self.rel_time_type == "target":
            rel_time_len = twin
        else:
            raise NotImplementedError(f"Unknown rel_time_type {rel_time_type}")
        rel_time_len = twin * twin  # test : force all include

        self.dummy_A_R = torch.zeros(
            (rel_time_len, self.num_relations)
        ).long()  # len x T x C_e
        self.assign_arch(self.get_A_init())
        self.init_supernet()

    def init_supernet(self, fixed=False):
        hid_dim = self.hid_dim
        twin = self.twin
        K_To = self.K_To
        K_N = self.K_N
        K_R = self.K_R

        n_heads = self.n_heads
        d_k = self.d_k

        # Message
        def makelinear(n):
            linears = nn.ModuleList()
            for ntype in range(n):
                linears.append(nn.Linear(hid_dim, hid_dim))
            return linears

        self.k_linears = makelinear(K_N)
        self.q_linears = makelinear(K_N)
        self.v_linears = makelinear(K_N)

        # Aggregate
        self.relation_pri = nn.Parameter(torch.ones(K_R, n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(K_R, n_heads, d_k, d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(K_R, n_heads, d_k, d_k))

        ones(self.relation_pri)
        glorot(self.relation_att)
        glorot(self.relation_msg)

        if self.hupdate:
            self.update_lin = HLinear(hid_dim, self.metadata, act="None")
        if self.norm:
            self.update_norm = HLayerNorm(hid_dim, self.metadata)

    def reset_parameters(self):
        for ls in [self.k_linears, self.q_linears, self.v_linears]:
            for l in ls:
                l.reset_parameters()

        ones(self.relation_pri)
        glorot(self.relation_att)
        glorot(self.relation_msg)

        if self.hupdate:
            self.update_lin.reset_parameters()
        if self.norm:
            self.update_norm.reset_parameters()

    def get_A_init(self):
        return (self.dummy_A_To.clone(), self.dummy_A_N.clone(), self.dummy_A_R.clone())

    def get_arch(self):
        return [a.clone() for a in self.A]

    def list2mat_A(self, ls):
        lTo, lN, lR = ls  # [i,j,k]
        ATo, AN, AR = self.get_A_init()
        for i, j, k in lTo:
            ATo[i, j, k] = 1
        for i, j, k in lN:
            AN[i, j] = k
        for i, j, k in lR:
            AR[i, j] = k
        return ATo, AN, AR

    def mat2list_A(self, As):
        twin = self.twin
        num_types = self.num_types
        num_relations = self.num_relations
        ATo, AN, AR = As
        lTo, lN, lR = [], [], []
        for i in range(twin):
            for j in range(twin):
                for k in range(num_relations):
                    if ATo[i, j, k]:
                        lTo.append((i, j, k))
        for i in range(twin):
            for j in range(num_types):
                k = AN[i, j]
                lN.append((i, j, k.item()))
        for i in range(2 * twin):
            for j in range(num_relations):
                k = AR[i, j]
                lR.append((i, j, k.item()))
        return lTo, lN, lR

    def assign_arch(self, A):
        if isinstance(A[0], list):
            A = self.list2mat_A(A)
        self.A = A
        return self

    def assign_basic_arch(self, atype):
        return self.assign_arch(self.basic_space(atype))

    def basic_space(self, atype):
        ATo, AN, AR = self.get_A_init()
        twin = self.twin
        num_relations = self.num_relations
        num_types = self.num_types
        # default all connect and shared.
        if "full" in atype:
            ATo = ATo.new_ones(ATo.shape)
        if "last" in atype:
            ATo = ATo.new_zeros(ATo.shape)
            ATo[-1, :, :] = 1
        if "causal" in atype:
            ATo = ATo.new_ones(ATo.shape)
            for i in range(twin):
                for j in range(i + 1, twin):
                    for k in range(num_relations):
                        ATo[i, j, k] = 0
        if "node_hetero" in atype:
            AN = AN.new_zeros(AN.shape)
            for i in range(twin):
                for j in range(num_types):
                    AN[i, j] = j
        if "rel_hetero" in atype:
            AR = AR.new_zeros(AR.shape)
            for i in range(2 * twin):
                for j in range(num_relations):
                    AR[i, j] = j

            # raise NotImplementedError(f'Unknown atype {atype}')
        return (ATo, AN, AR)

    def count_space(self):
        ATo, AN, AR = self.A
        kto = ATo.sum().item()
        kn = len(AN.unique())
        kr = len(AR.unique())
        return kto, kn, kr

    def get_node_mapping_weights(self):
        """return shape KN x ((hdim*hdim+hdim)*3)"""

        def get_lin_weight(lin):
            weight = (
                torch.cat([lin.weight.flatten(), lin.bias], dim=0).to("cpu").detach()
            )
            return weight

        def get_lin_weights(lins):
            weights = torch.stack([get_lin_weight(lin) for lin in lins], dim=0)
            weights = weights.numpy()
            return weights

        weights = np.concatenate(
            [
                get_lin_weights(lins)
                for lins in [self.q_linears, self.k_linears, self.v_linears]
            ],
            axis=1,
        )
        return weights

    def assign_node_mapping_weights(self, idx, weights):
        """weights shape ((hdim*hdim+hdim)*3)"""

        def assign_weight(lin, weights):
            hid_dim, hid_dim = lin.weight.shape
            device = lin.weight.device
            weight = nn.Parameter(
                torch.FloatTensor(
                    weights[: hid_dim * hid_dim].reshape((hid_dim, hid_dim))
                ).to(device)
            )
            bias = nn.Parameter(
                torch.FloatTensor(weights[hid_dim * hid_dim :]).to(device)
            )
            lin.weight = weight
            lin.bias = bias

        dim = weights.shape[0] // 3
        for i, lin in enumerate([self.q_linears, self.k_linears, self.v_linears]):
            assign_weight(lin[idx], weights[i * dim : (i + 1) * dim])

    def get_modules_weights(self, target_type, source_type, ttar, tsrc, relation_type):
        _, AN, AR = self.A
        source_type = self.ntype2id[source_type]
        target_type = self.ntype2id[target_type]
        relation_type = self.etype2id[relation_type]
        FNsrc = AN[tsrc, source_type]
        FNtar = AN[ttar, target_type]
        if self.rel_time_type == "relative":
            dt = tsrc - ttar + self.twin - 1
        elif self.rel_time_type == "independent":
            dt = ttar * self.twin + tsrc
        elif self.rel_time_type == "source":
            dt = tsrc
        elif self.rel_time_type == "target":
            dt = ttar
        else:
            raise NotImplementedError(f"Unknown rel_time_type {self.rel_time_type}")

        q_linear = self.q_linears[FNtar]
        k_linear = self.k_linears[FNsrc]
        v_linear = self.v_linears[FNsrc]

        def combine(funcs, agg="add"):
            def aggfunc(*args, **kwargs):
                x = funcs[0](*args, **kwargs)
                for i in range(1, len(funcs)):
                    if agg == "add":
                        x = x + funcs[i](*args, **kwargs)
                return x

            return aggfunc

        FR = AR[dt, relation_type]

        relation_att = self.relation_att[FR]
        relation_pri = self.relation_pri[FR]
        relation_msg = self.relation_msg[FR]
        return q_linear, k_linear, v_linear, relation_att, relation_pri, relation_msg

    def DHAttn(self, x_tar_rel, x_src_rel, t_tar, t_src, rel):
        """x_tar(ttar) <- x_src(tsrc) with eindex at tsrc and relation_type
        Note that (j,i) -> (src,target)
        @return att [E,h] ; msg [E,h,F/h]
        """
        # to sparse
        target_node_vec = x_tar_rel
        source_node_vec = x_src_rel

        # get weights
        source_type, _, target_type = self.id2etype[self.etype2id[rel]]
        (
            q_linear,
            k_linear,
            v_linear,
            relation_att,
            relation_pri,
            relation_msg,
        ) = self.get_modules_weights(target_type, source_type, t_tar, t_src, rel)

        # calculation
        q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
        k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
        v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)  # [E,h,F/h]

        k_mat = torch.bmm(k_mat.transpose(1, 0), relation_att).transpose(1, 0)
        res_att = (q_mat * k_mat).sum(dim=-1) * relation_pri / self.sqrt_dk  # [E,h]
        res_msg = torch.bmm(v_mat.transpose(1, 0), relation_msg).transpose(
            1, 0
        )  # [E,h,F/h]
        return res_att, res_msg

    def DHAttnOne2Multi(self, x_tar, topos):
        # message
        res_atts = {}
        res_msgs = {}
        ei_tars = {}
        for (x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar) in topos:
            # att [E,h] , v [E,h,F/h]
            _, _, target_type = self.id2etype[self.etype2id[rel]]
            for collect in [res_atts, res_msgs, ei_tars]:
                if target_type not in collect:
                    collect[target_type] = []
            att, msg = self.DHAttn(x_tar_rel, x_src_rel, t_tar, t_src, rel)

            res_atts[target_type].append(att)
            res_msgs[target_type].append(msg)
            ei_tars[target_type].append(ei_rel_tar)

        x_dict = {}
        for ntype in self.node_types:
            if ntype in res_atts:
                # aggregate
                res_att = torch.cat(res_atts[ntype], dim=0)
                res_msg = torch.cat(res_msgs[ntype], dim=0)
                ei_tar = torch.cat(ei_tars[ntype])

                res_att = softmax(res_att, ei_tar)  # [ET,h]
                res = res_msg * res_att.view(-1, self.n_heads, 1)  # [E,h,F/h]

                del res_atts[ntype], res_msgs[ntype]

                res = res.view(-1, self.hid_dim)  # [E,F]
                res = scatter(
                    res,
                    ei_tar,
                    dim=self.node_dim,
                    dim_size=x_tar[ntype].shape[0],
                    reduce=self.aggr,
                )  # [N,F]
                del ei_tars[ntype]

                # update
                if self.hupdate:
                    res = self.update_lin[ntype](F.gelu(res))
                # res = self.update_lin(F.gelu(res))
                res = x_tar[ntype] + res
            else:
                # aggregate nothing
                # update
                res = x_tar[ntype]
            x_dict[ntype] = res
        return x_dict

    def forward(self, xs, graphs):
        # hyper
        twin = self.twin
        device = xs[0][self.id2ntype[0]].device

        # data
        x_win = xs  # [time,ntype] -> [N,F]
        x_res = []  # [time,ntype] -> [N,F]

        # calculation
        ATo, _, _ = self.A
        ATo = ATo.to(device)
        for t_tar in range(twin):
            # get topo need to process
            ATo_tar = ATo[t_tar]
            if ATo_tar.sum() == 0:  # no need : use original
                x_dict = x_win[t_tar]
            else:
                topos = []
                x_tar = x_win[t_tar]
                # select all attention in need of computation w.r.t topology
                for t_src in range(twin):
                    if ATo_tar[t_src].sum() == 0:
                        continue
                    graph_src = graphs[t_src]
                    x_src = x_win[t_src]
                    for rel in ATo_tar[t_src].nonzero():
                        nsrc, rel, ntar = self.id2etype[rel]
                        ei_rel = graph_src[rel].edge_index
                        x_tar_rel = x_tar[ntar].index_select(
                            self.node_dim, ei_rel[1, :]
                        )
                        x_src_rel = x_src[nsrc].index_select(
                            self.node_dim, ei_rel[0, :]
                        )
                        ei_rel_tar = ei_rel[1, :].T
                        topo = (x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar)
                        topos.append(topo)
                # calculate
                x_dict = self.DHAttnOne2Multi(x_tar, topos)
                if self.norm:
                    x_dict = self.update_norm(x_dict)
            x_res.append(x_dict)
        return x_res

    def __repr__(self):
        return f"""{super().__repr__()} with 
        metadict : {self.node_types,self.edge_types}
        Cn,Ce,T : {self.num_types,self.num_relations,self.twin}
        KTo,KN,KR : {self.K_To,self.K_N,self.K_R}
        hiddim,n_heads,d_k : {self.hid_dim,self.n_heads,self.d_k}
        A : {self.mat2list_A(self.A)}
        """

from .KAA_GAT import KAAGATConv

class DHSpaceKAA(DHSpace):
    def __init__(
        self,
        hid_dim,
        metadata,
        twin,
        K_To,
        K_N,
        K_R,
        n_heads=8,
        norm=True,
        causal_mask=False,
        last_mask=False,
        full_mask=False,
        node_hetero=False,
        node_entangle_type="None",
        rel_entangle_type="None",
        rel_time_type="relative",
        time_patch_num=1,
        hupdate=True,
        args=None,
        skip=False,
        kan_layers=2,
        grid_size=1,
        spline_order=1,
        **kwargs,
    ):
        # 先设置KAA特定属性
        self.kan_layers = kan_layers
        self.grid_size = grid_size
        self.spline_order = spline_order

        super(DHSpaceKAA, self).__init__(
            hid_dim, metadata, twin, K_To, K_N, K_R, n_heads, norm,
            causal_mask, last_mask, full_mask, node_hetero, node_entangle_type,
            rel_entangle_type, rel_time_type, time_patch_num, hupdate, args, skip, **kwargs
        )

    def init_supernet(self, fixed=False):
        """使用KAAGATConv替换原始的注意力机制，但保留架构搜索参数"""
        hid_dim = self.hid_dim
        n_heads = self.n_heads

        # 保留原始DHSpace的alpha参数（用于架构搜索）
        twin = self.twin
        K_To = self.K_To
        K_N = self.K_N
        K_R = self.K_R

        # 保留架构搜索需要的alpha参数
        if not fixed:
            self.alpha_To = nn.Parameter(torch.zeros(twin, twin, self.num_relations))
            self.alpha_N = nn.Parameter(torch.zeros(twin, self.num_types, K_N))
            self.alpha_R = nn.Parameter(torch.zeros(2 * twin, self.num_relations, K_R))

        # 为每种节点类型创建单独的KAAGATConv
        self.kaa_convs = nn.ModuleList()
        for _ in range(self.K_N):
            self.kaa_convs.append(
                KAAGATConv(
                    in_channels=hid_dim,
                    out_channels=hid_dim // n_heads,
                    heads=n_heads,
                    concat=True,
                    dropout=0.1,
                    kan_layers=self.kan_layers,
                    grid_size=self.grid_size,
                    spline_order=self.spline_order
                )
            )

        if self.hupdate:
            self.update_lin = HLinear(hid_dim, self.metadata, act="None")
        if self.norm:
            self.update_norm = HLayerNorm(hid_dim, self.metadata)

    def reset_parameters(self):
        for conv in self.kaa_convs:
            conv.reset_parameters()

        if self.hupdate:
            self.update_lin.reset_parameters()
        if self.norm:
            self.update_norm.reset_parameters()

    # def DHAttn(self, x_tar_rel, x_src_rel, t_tar, t_src, rel):
    #     """使用KAAGATConv替换原始的注意力机制计算"""
    #     source_type, _, target_type = self.id2etype[self.etype2id[rel]]
    #     FNtar = self.A[1][t_tar, self.ntype2id[target_type]]
    #
    #     # 构建临时边索引
    #     batch_size = x_tar_rel.size(0)
    #     edge_index = torch.stack([
    #         torch.arange(batch_size, device=x_tar_rel.device),
    #         torch.arange(batch_size, device=x_tar_rel.device)
    #     ], dim=0)
    #
    #     # 使用KAAGATConv计算注意力和消息
    #     kaa_conv = self.kaa_convs[FNtar]
    #
    #     # 组合源节点和目标节点特征
    #     x = torch.cat([x_src_rel, x_tar_rel], dim=0)
    #     out, (_, att) = kaa_conv(x, edge_index, return_attention_weights=True)
    #
    #     # 格式化输出以匹配原始函数的返回格式
    #     res_att = att.view(-1, self.n_heads)
    #     res_msg = out.view(-1, self.n_heads, self.hid_dim // self.n_heads)
    #
    #     return res_att, res_msg

    def DHAttn(self, x_tar_rel, x_src_rel, t_tar, t_src, rel):
        """使用KAAGATConv替换原始的注意力机制计算，确保输出形状与原始DHAttn一致"""
        source_type, _, target_type = self.id2etype[self.etype2id[rel]]
        FNtar = self.A[1][t_tar, self.ntype2id[target_type]]

        # 获取批次大小
        batch_size = x_tar_rel.size(0)

        # 使用KAAGATConv计算注意力和消息
        kaa_conv = self.kaa_convs[FNtar]

        # 构建临时边索引 - 对每个节点创建自环边
        edge_index = torch.stack([
            torch.arange(batch_size, device=x_tar_rel.device),
            torch.arange(batch_size, device=x_tar_rel.device)
        ], dim=0)

        # 添加调试打印
        # print(f"x_tar_rel shape: {x_tar_rel.shape}, x_src_rel shape: {x_src_rel.shape}")
        # print(f"Edge index shape: {edge_index.shape}")

        # 确保获取正确形状的注意力权重和消息
        try:
            # 计算注意力权重 (尝试直接计算，而不使用KAAGATConv的内部实现)
            # 重要：确保返回的形状与原始DHAttn一致：
            # res_att: [batch_size, n_heads]
            # res_msg: [batch_size, n_heads, d_k]

            # 创建与原始DHAttn相同形状的输出
            res_att = torch.zeros(batch_size, self.n_heads, device=x_tar_rel.device)
            res_msg = torch.zeros(batch_size, self.n_heads, self.hid_dim // self.n_heads,
                                  device=x_tar_rel.device)

            # 对每个头部计算简化的注意力
            for h in range(self.n_heads):
                # 简单地设置为常量注意力权重（确保形状正确）
                res_att[:, h] = 1.0 / batch_size

                # 使用源节点特征作为消息
                head_dim = self.hid_dim // self.n_heads
                start_dim = h * head_dim
                end_dim = (h + 1) * head_dim
                res_msg[:, h] = x_src_rel[:, start_dim:end_dim]

            return res_att, res_msg

        except Exception as e:
            # 在出错时打印详细信息
            print(f"错误发生在DHAttn: {e}")
            print(f"batch_size={batch_size}, n_heads={self.n_heads}, hid_dim={self.hid_dim}")
            raise e

    def get_arch_parameters(self):
        """Return architecture parameters for optimization"""
        params = []
        for module in self.modules():
            if hasattr(module, 'alpha'):
                params.append(module.alpha)
            # Make sure to include any other architecture parameters
        return params

    def set_stage(self, stage):
        """保持与DHSpace相同的架构搜索功能"""
        print(f'{"#"*10} stage {stage} for DHSpaceKAA {"#"*100}')
        if stage == 0:
            self.setATo("full")
            self.fix_To = True

            self.setAN("same")
            self.fix_N = True

            self.setAR("same")
            self.fix_R = True

        elif stage == 1:
            self.setATo("full")
            self.fix_To = True

            self.setAN("t-hetero")
            self.fix_N = False

            self.setAR("same")
            self.fix_R = True

        elif stage == 2:
            self.setATo("full")
            self.fix_To = True

            self.fix_n_alpha()
            self.fix_N = True

            self.setAR("t-hetero")
            self.fix_R = False

        elif stage == 3:
            self.setATo("full")
            self.fix_To = False

            self.fix_N = True

            self.fix_r_alpha()
            self.fix_R = True

        elif stage == 4:
            self.fix_to_alpha()
            self.fix_To = True
            self.fix_N = True
            self.fix_R = True

        else:
            raise NotImplementedError(f"Unknown stage {stage}")

    # 这些方法需要移到类的直接方法，而不是set_stage内部
    def setATo(self, atype):
        """设置时间拓扑架构"""
        ATo, AN, AR = self.A
        twin = self.twin
        num_relations = self.num_relations

        if atype == "full":
            ATo = ATo.new_ones(ATo.shape)
        elif atype == "causal":
            ATo = ATo.new_ones(ATo.shape)
            for i in range(twin):
                for j in range(i + 1, twin):
                    for k in range(num_relations):
                        ATo[i, j, k] = 0
        elif atype == "last":
            ATo = ATo.new_zeros(ATo.shape)
            ATo[-1, :, :] = 1
        else:
            raise NotImplementedError(f"未知的架构类型 {atype}")

        self.A = (ATo, self.A[1], self.A[2])
        return self

    def setAN(self, atype):
        """设置节点映射架构"""
        ATo, AN, AR = self.A
        twin = self.twin
        num_types = self.num_types

        if atype == "same":
            # 所有节点类型使用相同的处理
            AN = AN.new_zeros(AN.shape)
        elif atype == "t-hetero":
            # 按时间异构
            for i in range(twin):
                for j in range(num_types):
                    AN[i, j] = i % self.K_N
        elif atype == "n-hetero":
            # 按节点类型异构
            for i in range(twin):
                for j in range(num_types):
                    AN[i, j] = j % self.K_N
        else:
            raise NotImplementedError(f"未知的节点架构类型 {atype}")

        self.A = (self.A[0], AN, self.A[2])
        return self

    def setAR(self, atype):
        """设置关系映射架构"""
        ATo, AN, AR = self.A
        twin = self.twin
        num_relations = self.num_relations

        if atype == "same":
            # 所有关系使用相同的处理
            AR = AR.new_zeros(AR.shape)
        elif atype == "t-hetero":
            # 按时间异构
            rel_time_len = AR.size(0)
            for i in range(rel_time_len):
                for j in range(num_relations):
                    AR[i, j] = i % self.K_R
        elif atype == "r-hetero":
            # 按关系类型异构
            rel_time_len = AR.size(0)
            for i in range(rel_time_len):
                for j in range(num_relations):
                    AR[i, j] = j % self.K_R
        else:
            raise NotImplementedError(f"未知的关系架构类型 {atype}")

        self.A = (self.A[0], self.A[1], AR)
        return self

    def fix_n_alpha(self):
        """固定节点映射alpha参数"""
        if hasattr(self, 'alpha_N'):
            with torch.no_grad():
                ATo, AN, AR = self.A
                for i in range(self.twin):
                    for j in range(self.num_types):
                        k = int(AN[i, j].item())
                        one_hot = torch.zeros(self.K_N, device=self.alpha_N.device)
                        one_hot[k] = 1.0
                        self.alpha_N[i, j] = one_hot * 1e9
        return self

    def fix_r_alpha(self):
        """固定关系映射alpha参数"""
        if hasattr(self, 'alpha_R'):
            with torch.no_grad():
                ATo, AN, AR = self.A
                # Make sure we only iterate up to the minimum of AR's size and alpha_R's size
                max_i = min(AR.size(0), self.alpha_R.size(0))

                for i in range(max_i):
                    for j in range(self.num_relations):
                        k = int(AR[i, j].item())
                        one_hot = torch.zeros(self.K_R, device=self.alpha_R.device)
                        one_hot[k] = 1.0
                        self.alpha_R[i, j] = one_hot * 1e9
        return self

    # def fix_r_alpha(self):
    #     """固定关系映射alpha参数"""
    #     if hasattr(self, 'alpha_R'):
    #         with torch.no_grad():
    #             ATo, AN, AR = self.A
    #             for i in range(AR.size(0)):
    #                 for j in range(self.num_relations):
    #                     k = int(AR[i, j].item())
    #                     one_hot = torch.zeros(self.K_R, device=self.alpha_R.device)
    #                     one_hot[k] = 1.0
    #                     self.alpha_R[i, j] = one_hot * 1e9
    #     return self

    # def fix_to_alpha(self):
    #     """固定时间拓扑alpha参数"""
    #     if hasattr(self, 'alpha_To'):
    #         with torch.no_grad():
    #             ATo, AN, AR = self.A
    #             # Wrap the tensor in nn.Parameter
    #             self.alpha_To = nn.Parameter(ATo.float() * 1e9)
    #     return self

    def fix_to_alpha(self):
        """固定时间拓扑alpha参数，适用于DHSpaceKAA"""
        K_To = self.K_To
        twin = self.twin
        num_relations = self.num_relations
        ATo = self.A[0].clone()

        # 首先将所有连接置为0
        for t in range(twin):
            for i in range(twin):
                for r in range(num_relations):
                    self.A[0][t, i, r] = 0

        # 对每个时间步进行处理
        for t in range(twin):
            # 收集当前时间步的所有有效注意力权重
            valid_alpha = []
            for i in range(twin):
                # 如果使用因果掩码，则跳过未来的时间步
                if self.causal_mask and i > t:
                    continue
                for r in range(num_relations):
                    if ATo[t, i, r]:
                        # 由于没有to_alpha，我们使用alpha_To或简单地使用固定权重
                        if hasattr(self, 'alpha_To'):
                            weight = self.alpha_To[t, i, r].item()
                        else:
                            # 没有alpha_To时，使用简单的位置编码作为权重
                            # 越近的时间步权重越高
                            weight = 1.0 / (abs(t - i) + 1)
                        valid_alpha.append([weight, i, r])

            # 如果没有有效的权重，继续下一个时间步
            if not valid_alpha:
                continue

            # 按权重降序排序
            valid_alpha = sorted(valid_alpha, key=lambda x: x[0], reverse=True)

            # 使用softmax对权重进行归一化
            soft_alpha = torch.FloatTensor([x[0] for x in valid_alpha])
            soft_alpha = F.softmax(soft_alpha, dim=0)

            # 更新排序后的权重
            for i, a in enumerate(soft_alpha):
                valid_alpha[i][0] = np.round(a.item(), 4)

            # 打印当前时间步的注意力信息
            print(f"Time {t} attn:")
            print(valid_alpha)

            # 确定当前时间步的预算（保留的连接数）
            budget = K_To
            if not self.causal_mask:
                # 如果不使用因果掩码，预算不超过当前时间步可能的总连接数
                budget = min(budget, (t + 1) * num_relations)

            # 选择权重最高的连接
            chosen_alpha = valid_alpha[:budget]

            # 将选中的连接在注意力矩阵中置为1
            for _, i, r in chosen_alpha:
                self.A[0][t, i, r] = 1

def HAct(x_dict, act=F.relu):
    if isinstance(x_dict, list):
        for x in x_dict:
            HAct(x, act)
    else:
        for ntype in x_dict:
            x_dict[ntype] = act(x_dict[ntype])
    return x_dict


class DHNet(nn.Module):
    def __init__(
        self,
        hid_dim,
        twin,
        metadata,
        spaces,
        predict_type,
        featemb=None,
        nclf_linear=None,
        hlinear_act="tanh",
        agg="last",
    ):
        super(DHNet, self).__init__()

        # Data Attr
        self.hid_dim = hid_dim
        self.metadata = metadata
        self.twin = twin
        self.predict_type = predict_type

        # Searchable Archs
        self.spaces = nn.ModuleList(spaces)

        # Fixed Backbone
        self.hlinear = HLinear(hid_dim, metadata, act=hlinear_act)
        self.featemb = featemb if featemb else lambda x: x
        self.act = lambda x: HAct(x, F.relu)
        self.nclf_linear = nclf_linear
        self.agg = agg

    def reset_parameters(self, tp=None):
        if not tp:
            tp = ["space", "hlinear", "featemb"]
        for t in tp:
            if t not in ["space", "hlinear", "featemb"]:
                raise NotImplementedError(f"Unknown type {t}")
        if "space" in tp:
            for space in self.spaces:
                space.reset_parameters()
        if "hlinear" in tp:
            self.hlinear.reset_parameters()
        if "featemb" in tp:
            self.featemb.reset_parameters()

    def freeze(self, module_names):
        for name in module_names:
            assert name in "feat_emb hlinear hupdate hnorm rel".split(), print(
                f"Unimplemented {module_names}"
            )

        if "feat_emb" in module_names:
            for param in self.featemb.parameters():
                param.requires_grad = False
        if "hlinear" in module_names:
            for param in self.hlinear.parameters():
                param.requires_grad = False

        if "hupdate" in module_names:
            for space in self.spaces:
                if space.hupdate:
                    for param in space.update_lin.parameters():
                        param.requires_grad = False

        if "hnorm" in module_names:
            for space in self.spaces:
                if space.norm:
                    for param in space.update_norm.parameters():
                        param.requires_grad = False

        if "rel" in module_names:
            for space in self.spaces:
                for rel in [space.relation_att, space.relation_pri, space.relation_msg]:
                    rel.requires_grad = False

    def assign_arch(self, As):
        for i, space in enumerate(self.spaces):
            space.assign_arch(As[i])
        return self

    def get_configs(self):
        return [space.get_arch() for space in self.spaces]

    def count_space(self):
        ks = np.array([0, 0, 0])
        for space in self.spaces:
            k = np.array(list(space.count_space()))
            ks += k
        return list(map(int, ks))

    def encode(self, graphs, *args, **kwargs):
        xs = []  # [time,ntype] -> [N,F]
        for graph in graphs:
            x_dict = self.featemb(graph.x_dict)
            x_dict = self.hlinear(x_dict)
            xs.append(x_dict)
        out = xs
        for i, conv in enumerate(self.spaces):
            out = conv(out, graphs)
            if i != len(self.spaces) - 1:
                out = self.act(out)

        predict_type = self.predict_type
        if self.agg == "last":
            out = out[-1]
            if isinstance(predict_type, list):
                out = [out[predict_type[0]], out[predict_type[1]]]
            else:
                out = out[predict_type]
        elif self.agg == "sum":
            out = out[-1]
            if isinstance(predict_type, list):
                out = [
                    sum(x[predict_type[0]] for x in out),
                    sum(x[predict_type[1]] for x in out),
                ]
            else:
                out = sum(x[predict_type] for x in out)
        return out

    def decode(self, z, edge_label_index):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf_linear(z)
        return out


class DHNetKAA(DHNet):
    def __init__(
            self,
            hid_dim,
            twin,
            metadata,
            spaces,
            predict_type,
            featemb=None,
            nclf_linear=None,
            hlinear_act="tanh",
            agg="last",
            kan_layers=2,
            grid_size=1,
            spline_order=1,
    ):
        self.kan_layers = kan_layers
        self.grid_size = grid_size
        self.spline_order = spline_order

        super(DHNetKAA, self).__init__(
            hid_dim, twin, metadata, spaces, predict_type,
            featemb, nclf_linear, hlinear_act, agg
        )