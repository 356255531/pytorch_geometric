import torch


class Net(torch.nn.Module):
    """
    abstract GCN models without a transformation pretrain
    """
    def __init__(self, emb_dim, num_nodes):
        super(Net, self).__init__()
        self.emb_dim = emb_dim

        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1, norm_type=2.0)

    def proj_node(self, x, edge_attr):
        """
        May need projection to get user or item embedding
        :param x:
        :return:
        """
        raise NotImplementedError('Entitiy projection operation for interaction prediction not defined!')

    def check_interact_edge(self, edge_attr):
        """
        Check if the given edge is interaction edges
        :param edge_attr:
        :return:
        """
        raise NotImplementedError('interaction edge check function not defined!')

    def predict(self, train_edge_index, train_interact_edge_index, train_interact_edge_attr, *args):
        self.check_interact_edge(train_interact_edge_attr)
        x = self(train_edge_index, *args)

        head = self.proj_node(x[train_interact_edge_index[:1, :].t()], train_interact_edge_attr)
        tail = self.proj_node(x[train_interact_edge_index[1:2, :].t()], train_interact_edge_attr)

        est_feedback = torch.sum(head * tail, dim=-1).reshape(-1, 1)
        return est_feedback


class KGNet(Net):
    """
    abstract GCN models with a transformation pretrain
    """
    def __init__(self, emb_dim, repr_dim, num_nodes, num_relations, pretrain):
        super(KGNet, self).__init__(emb_dim, num_nodes)
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim
        self.num_relations = num_relations
        self.pretrain = pretrain

        self.r_emb = torch.nn.Embedding(
            num_relations, emb_dim, max_norm=1, norm_type=2.0
        )
        if not self.pretrain:
            raise ValueError('Use Net super class instead of KGNet!')
        if self.pretrain == 'trans_e':
            if self.emb_dim != self.repr_dim:
                raise ValueError('TransE must have same representation and embedding dimension!')
        elif self.pretrain == 'trans_r':
            self.r_proj = torch.nn.Embedding(
                self.num_relations // 2, self.emb_dim * self.repr_dim, max_norm=1, norm_type=2.0
            )
        elif self.pretrain == 'trans_h':
            self.r_proj = torch.nn.Embedding(
                self.num_relations // 2, self.emb_dim, max_norm=1, norm_type=2.0
            )
            if self.emb_dim != self.repr_dim:
                raise ValueError('TransH must have same representation and embedding dimension!')
        else:
            raise NotImplementedError('Pretain: {} not implemented!'.format(self.pretrain))

        self.kg_loss_func = torch.nn.MSELoss()

    def proj_kg_node(self, x, edge_attr):
        if self.pretrain == 'trans_e':
            proj_kg_node = self.trans_e_project
        elif self.pretrain == 'trans_r':
            proj_kg_node = self.trans_r_project
        elif self.pretrain == 'trans_h':
            proj_kg_node = self.trans_h_project
        else:
            raise NotImplementedError('Pretain: {} not implemented!'.format(self.pretrain))

        return proj_kg_node(x, edge_attr)

    def get_kg_loss(self, edge_index, edge_attr):
        r_idx = edge_attr[:, 0]
        r_emb = self.r_emb.weight[r_idx]

        head = self.proj_kg_node(self.node_emb.weight[edge_index[:1, :].t()], edge_attr)
        tail = self.proj_kg_node(self.node_emb.weight[edge_index[1:2, :].t()], edge_attr)

        loss_t = self.kg_loss_func(head + r_emb, tail)

        return loss_t

    def trans_e_project(self, node_emb, edge_attr):
        return node_emb

    def trans_r_project(self, node_emb, edge_attr):
        r_idx = edge_attr[:, 0]

        r_proj = self.r_proj.weight[r_idx // 2].reshape(-1, self.emb_dim, self.emb_dim)
        proj_node = torch.matmul(node_emb, r_proj).reshape(-1, self.emb_dim)
        return proj_node

    def get_trans_d_loss(self, node_emb, edge_attr):
        r_idx = edge_attr[:, 0]

        r_proj = self.r_proj.weight[r_idx // 2]
        proj_node = node_emb - torch.matmul(torch.matmul(r_proj, node_emb), r_proj.t())
        return proj_node
