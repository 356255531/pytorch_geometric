import torch


class Net(torch.nn.Module):
    def __init__(self, emb_dim, num_nodes):
        super(Net, self).__init__()
        self.emb_dim = emb_dim

        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1, norm_type=2.0)

    def project_node(self, x):
        """
        May need projection to get user or item embedding
        :param x:
        :return:
        """
        raise NotImplementedError('Entitiy projection operation not defined!')

    def predict(self, train_edge_index, train_interact_edge_index, *args):
        x = self(train_edge_index, *args)
        head = self.project_node(x[train_interact_edge_index[0, :]])
        tail = self.project_node(x[train_interact_edge_index[1, :]])

        est_feedback = torch.sum(head * tail, dim=1).reshape(-1, 1)
        return est_feedback


class KGNet(Net):
    def __init__(self, emb_dim, num_nodes, num_relations, pretrain='trans_e'):
        super(KGNet, self).__init__(emb_dim, num_nodes)
        self.emb_dim = emb_dim
        self.r_emb = torch.nn.Embedding(
            num_relations, emb_dim, max_norm=1, norm_type=2.0
        )
        self.kg_loss_func = torch.nn.MSELoss()

        if pretrain == 'trans_e':
            self.proj_kg_node = self.trans_e_project
        elif pretrain == 'trans_r':
            self.r_proj = torch.nn.Embedding(
                num_relations // 2, emb_dim * emb_dim, max_norm=1, norm_type=2.0
            )
            self.proj_kg_node = self.trans_r_project
        elif pretrain == 'trans_h':
            self.r_proj = torch.nn.Embedding(
                num_relations // 2, emb_dim, max_norm=1, norm_type=2.0
            )
            self.proj_kg_node = self.trans_h_project
        else:
            raise NotImplementedError('Pretain: {} not implemented!'.format(pretrain))

    def get_kg_loss(self, edge_index_t, edge_attr):
        r_idx = edge_attr[:, 0]
        r_emb = self.r_emb.weight[r_idx]

        r_idx /= 2
        head = self.proj_kg_node(self.node_emb.weight[edge_index_t[:1, :]], r_idx)
        tail = self.proj_kg_node(self.node_emb.weight[edge_index_t[1:2, :]], r_idx)

        loss_t = self.kg_loss_func(head + r_emb, tail.detach())

        return loss_t

    def trans_e_project(self, node_emb, r_idx):
        return node_emb

    def trans_r_project(self, node_emb, r_idx):
        r_proj = self.r_proj.weight[r_idx]
        proj_node = torch.matmul(node_emb, r_proj).reshape(-1, self.emb_dim)
        return proj_node

    def get_trans_d_loss(self, node_emb, r_idx):
        r_proj = self.r_proj.weight[r_idx]
        proj_node = node_emb - torch.matmul(torch.matmul(r_proj, node_emb), r_proj.t())
        return proj_node
