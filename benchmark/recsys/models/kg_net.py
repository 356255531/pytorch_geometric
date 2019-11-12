import torch


class KGNet(torch.nn.Module):
    def __init__(self, emb_dim, repr_dim, num_nodes, num_relations):
        super(KGNet, self).__init__()
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim

        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1, norm_type=2.0)
        self.r_emb = torch.nn.Embedding(num_relations, repr_dim, max_norm=1, norm_type=2.0)
        self.r_proj = torch.nn.Embedding(
            num_relations, emb_dim * repr_dim, max_norm=1, norm_type=2.0
        )

        self.kg_loss_func = torch.nn.MSELoss()

    def predict(self, edge_index, train_trans_edge_index, **kwargs):
        x = self(edge_index, **kwargs)
        head = x[train_trans_edge_index[:, 0]]
        tail = x[train_trans_edge_index[:, 1]]

        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        return est_rating

    def get_kg_loss(self, edge_index, edge_attr):
        r_idx = edge_attr[:, 0]
        r_emb = self.r_emb.weight[r_idx]
        r_proj = self.r_proj.weight[r_idx].reshape(-1, self.emb_dim, self.repr_dim)
        proj_head = torch.matmul(self.node_emb.weight[edge_index[:, :1]], r_proj).reshape(-1, self.repr_dim)

        proj_tail = torch.matmul(self.node_emb.weight[edge_index[:, 1:2]], r_proj).reshape(-1, self.repr_dim)

        est_tail = proj_head + r_emb

        loss_t = self.kg_loss_func(est_tail, proj_tail)

        return loss_t
