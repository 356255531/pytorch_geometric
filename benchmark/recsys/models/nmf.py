import torch


class NMF(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_mf, hidden_mlp, layers):
        super(NMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_mf = hidden_mf
        self.latent_dim_mlp = hidden_mlp

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1] + hidden_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.embedding_user_mlp.weight, -1, 1)
        torch.nn.init.normal_(self.embedding_user_mf.weight, -1, 1)
        torch.nn.init.normal_(self.embedding_item_mf.weight, -1, 1)
        torch.nn.init.normal_(self.embedding_item_mf.weight, -1, 1)
        for layer in self.fc_layers:
            torch.nn.init.normal_(layer.weight, -1, 1)
        torch.nn.init.normal_(self.affine_output.weight, -1, 1)

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

