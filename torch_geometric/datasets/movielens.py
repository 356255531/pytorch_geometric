import torch
from collections import defaultdict
from torch_geometric.data import InMemoryDataset, download_url
from os.path import join
import numpy as np
import tqdm
import pickle

from torch_geometric.io import read_ml
from torch_geometric.data import Data, extract_zip


def reindex_df(users, items, interactions):
    """
    reindex users, items, interactions in case there are some values missing in between
    :param users: pd.DataFrame
    :param items: pd.DataFrame
    :param interactions: pd.DataFrame
    :return: same
    """
    n_users = users.shape[0]
    n_movies = items.shape[0]

    raw_uids = np.array(users.uid, dtype=np.int)
    raw_iids = np.array(items.iid, dtype=np.int)
    uids = np.arange(n_users)
    iids = np.arange(n_movies)

    users.loc[:, 'uid'] = uids
    items.loc[:, 'iid'] = iids

    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(raw_uids, uids)}
    raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(raw_iids, iids)}

    rating_uids = np.array(interactions.uid, dtype=np.int)
    rating_iids = np.array(interactions.iid, dtype=np.int)
    print('reindex user id of ratings...')
    for uid in tqdm.tqdm(raw_uids):
        rating_uids[rating_uids == uid] = raw_uid2uid[uid]
    print('reindex item id of ratings...')
    for iid in tqdm.tqdm(raw_iids):
        rating_iids[rating_iids == iid] = raw_iid2iid[iid]
    interactions.loc[:, 'uid'] = rating_uids
    interactions.loc[:, 'iid'] = rating_iids

    return users, items, interactions


def convert_2_data(users, items, ratings, emb_dim, repr_dim, split):
    """
    Entitiy node include (gender, occupation, genres)

    n_nodes = n_users + n_items + n_genders + n_occupation + n_genres

    """
    n_users = users.shape[0]
    n_items = items.shape[0]
    n_interactions = ratings.shape[0]

    genders = ['M', 'F']
    n_genders = len(genders)
    gender_map = {r: i for i, r in enumerate(genders)}

    occupations = list(users.occupation.unique())
    n_occupations = len(occupations)
    occupation_map = {r: i for i, r in enumerate(occupations)}

    n_genres = 18

    relations = ['gender', 'occupation', 'genre', 'interact', '-gender', '-occupation', '-genre', '-interact']
    n_relations = len(relations)
    relation_map = {r: i for i, r in enumerate(relations)}

    n_nodes = n_users + n_items + n_genders + n_occupations + n_genres

    x = torch.nn.Embedding(n_nodes, emb_dim, max_norm=1, norm_type=2.0).weight

    r_emb = torch.nn.Embedding(n_relations, repr_dim, max_norm=1, norm_type=2.0).weight
    r_proj = torch.nn.Embedding(n_relations // 2, emb_dim * repr_dim, max_norm=1, norm_type=2.0).weight
    r_proj = torch.cat((r_proj, -r_proj), dim=0)

    row_idx, col_idx = [], []
    edge_attrs = []

    rating_begin = 0

    print('Creating user property edges...')
    for _, row in tqdm.tqdm(users.iterrows(), total=users.shape[0]):
        gender = row['gender']
        occupation = row['occupation']

        u_nid = row['uid']
        gender_nid = n_users + n_items + gender_map[gender]
        row_idx.append(u_nid)
        col_idx.append(gender_nid)
        edge_attrs.append([relation_map['gender'], -1])

        occupation_nid = n_users + n_items + n_genders + occupation_map[occupation]
        row_idx.append(u_nid)
        col_idx.append(occupation_nid)
        edge_attrs.append([relation_map['occupation'], -1])
    rating_begin += 2 * users.shape[0]

    print('Creating item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        i_nid = n_users + row['iid']

        genres = list(row[3:])

        for idx, genre in enumerate(genres):
            if not genre:
                continue
            g_nid = n_users + n_items + n_genders + n_occupations + idx
            row_idx.append(i_nid)
            col_idx.append(g_nid)
            edge_attrs.append([relation_map['genre'], -1])
            rating_begin += 1

    print('Creating rating property edges...')
    for _, row in tqdm.tqdm(ratings.iterrows(), total=ratings.shape[0]):
        u_nid = row['uid']
        i_nid = row['iid']

        row_idx.append(u_nid)
        col_idx.append(i_nid)
        edge_attrs.append([relation_map['interact'], row['rating']])

    half_edge_mask = torch.zeros((rating_begin, 1))
    rating_mask = torch.ones((ratings.shape[0], 1))
    rating_edge_mask = torch.cat((half_edge_mask, rating_mask, half_edge_mask, rating_mask), dim=0)
    if split is not None:
        pass


    print('Creating reverse user property edges...')
    for _, row in tqdm.tqdm(users.iterrows(), total=users.shape[0]):
        gender = row['gender']
        occupation = row['occupation']

        u_nid = row['uid']
        gender_nid = n_users + n_items + gender_map[gender]
        col_idx.append(u_nid)
        row_idx.append(gender_nid)
        edge_attrs.append([relation_map['-gender'], -1])

        occupation_nid = n_users + n_items + n_genders + occupation_map[occupation]
        col_idx.append(u_nid)
        row_idx.append(occupation_nid)
        edge_attrs.append([relation_map['-occupation'], -1])

    print('Creating reverse item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        i_nid = n_users + row['iid']

        genres = list(row[3:])

        for idx, genre in enumerate(genres):
            if not genre:
                continue
            g_nid = n_users + n_items + n_genders + n_occupations + idx
            col_idx.append(i_nid)
            row_idx.append(g_nid)
            edge_attrs.append([relation_map['-genre'], -1])

    print('Creating reverse rating property edges...')
    for _, row in tqdm.tqdm(ratings.iterrows(), total=ratings.shape[0]):
        u_nid = row['uid']
        i_nid = row['iid']

        col_idx.append(u_nid)
        row_idx.append(i_nid)
        edge_attrs.append([relation_map['-interact'], row['rating']])

    row_idx = np.array(row_idx).reshape(1, -1)
    col_idx = np.array(col_idx).reshape(1, -1)
    edge_index = np.concatenate((row_idx, col_idx), axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    edge_attrs = np.array(edge_attrs)
    edge_attrs = torch.from_numpy(edge_attrs).long()

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attrs,
        r_emb=r_emb, r_proj=r_proj, n_users=n_users,
        n_items=n_items, n_interactions=n_interactions, relation_map=relation_map
    )


def random_split(edge_index, splits):
    print('Splitting data...')
    prior_r, train_idx, test_r = splits

    n_edges = edge_index.shape[1]
    edge_idx_idx = np.arange(n_edges)

    np.random.seed(1234)
    np.random.shuffle(edge_idx_idx)

    prior_upper_bound = int(n_edges * prior_r)
    train_upper_bound = int(n_edges * (prior_r + train_idx))
    prior_edge_idx_idx = edge_idx_idx[:prior_upper_bound]
    train_edge_idx_idx = edge_idx_idx[prior_upper_bound: train_upper_bound]
    test_edge_idx_idx = edge_idx_idx[train_upper_bound:]

    return prior_edge_idx_idx, train_edge_idx_idx, test_edge_idx_idx


def create_adj_dict(edge_idx, edge_attr, prior_edge_idx_idx):
    print('Creating adjacency dictionary...')
    rows, cols = edge_idx[0, :], edge_idx[1, :]
    rs = edge_attr[:, 0]
    adj_dict = defaultdict(list)
    for row, col, r in zip(rows, cols, rs):
        adj_dict[row].append((r, col))
        adj_dict[col].append((r, row))

    rows, cols = edge_idx[0, prior_edge_idx_idx], edge_idx[1, prior_edge_idx_idx]
    rs = edge_attr[prior_edge_idx_idx, 0]
    prior_adj_dict = defaultdict(list)
    for row, col, r in zip(rows, cols, rs):
        prior_adj_dict[row].append((r, col))
        prior_adj_dict[col].append((r, row))

    return adj_dict, prior_adj_dict


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def restore(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


class MovieLens(InMemoryDataset):
    url = 'http://files.grouplens.org/datasets/movielens/'

    def __init__(self,
                 root,
                 name,
                 emb_dim=300,
                 repr_dim=64,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 debug=False):
        self.name = name.lower()
        assert self.name in ['1m']
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim
        self.debug = debug
        super(MovieLens, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        # if self.splits is not None:
        #     self.edge_idx = np.load(self.processed_paths[1])
        #     self.edge_attrs = np.load(self.processed_paths[2])
        #     self.prior_edge_idx_idx = np.load(self.processed_paths[3])
        #     self.train_edge_idx_idx = np.load(self.processed_paths[4])
        #     self.test_edge_idx_idx = np.load(self.processed_paths[5])
        #
        #     [self.n_users, self.n_items, self.n_interactions, self.n_nodes, self.n_edges] = restore(self.processed_paths[6])
        #     self.adj_dict = restore(self.processed_paths[7])
        #     self.prior_adj_dict = restore(self.processed_paths[8])
        print('Dataset loaded!')

    @property
    def raw_file_names(self):
        return 'ml-{}.zip'.format(self.name.lower())

    @property
    def processed_file_names(self):
        return ['data.pt']
        # return [
        #     'data.pt', 'edge_idx.npy', 'edge_attrs.npy', 'prior_edge_idx_idx.npy', 'train_edge_idx_idx.npy',
        #     'test_edge_idx_idx.npy', 'statistics.pkl', 'adj_dict.pkl', 'prior_adj_dict.pkl']

    def download(self):
        path = download_url(self.url + self.raw_file_names, self.raw_dir)

        extract_zip(path, self.raw_dir)

    # def gen_save_ds_property(self, users, items, ratings, splits):
    #     """
    #     Only include each edge for one time in edge_idx
    #     :param users: users df
    #     :param items: items df
    #     :param ratings: ratings df
    #     :param splits: str, '0.8,0.1,0.1'
    #     :return:
    #     """
    #     n_users = users.shape[0]
    #     n_items = items.shape[0]
    #     n_interactions = ratings.shape[0]
    #     n_nodes = n_users + n_items
    #     n_edges = n_interactions
    #
    #     row_idx, col_idx = [], []
    #     edge_attrs = []
    #     print('Creating dataset property...')
    #     for _, row in tqdm.tqdm(ratings.iterrows(), total=ratings.shape[0]):
    #         u_nid = row['uid']
    #         i_nid = row['iid'] + n_users
    #         r = row['rating']
    #         t = row['timestamp']
    #
    #         row_idx.append(u_nid)
    #         col_idx.append(i_nid)
    #         edge_attrs.append([r, t])
    #
    #     edge_idx = np.concatenate((np.array(row_idx).reshape(1, -1), np.array(col_idx).reshape(1, -1)), axis=0)
    #     edge_attrs = np.array(edge_attrs)
    #
    #     prior_edge_idx_idx, train_edge_idx_idx, test_edge_idx_idx = random_split(edge_idx, splits=splits)
    #     adj_dict, prior_adj_dict = create_adj_dict(edge_idx, edge_attrs, prior_edge_idx_idx)
    #
    #     np.save(self.processed_paths[1], edge_idx)
    #     np.save(self.processed_paths[2], edge_attrs)
    #     np.save(self.processed_paths[3], prior_edge_idx_idx)
    #     np.save(self.processed_paths[4], train_edge_idx_idx)
    #     np.save(self.processed_paths[5], test_edge_idx_idx)
    #     save([n_users, n_items, n_interactions, n_nodes, n_edges], self.processed_paths[6])
    #     save(adj_dict, self.processed_paths[7])
    #     save(prior_adj_dict, self.processed_paths[8])

    def process(self):
        unzip_raw_dir = join(self.raw_dir, 'ml-{}'.format(self.name))

        # read files
        users, items, ratings = read_ml(unzip_raw_dir, self.debug)

        item_count = ratings['iid'].value_counts()
        user_count = ratings['uid'].value_counts()
        item_count.name = 'movie_count'
        user_count.name = 'user_count'
        ratings = ratings.join(item_count, on='iid')
        ratings = ratings.join(user_count, on='uid')

        # delete ratings have movie and user count less than 10
        ratings = ratings[ratings.movie_count > 9]
        ratings = ratings[ratings.user_count > 9]

        # drop users and movies which do not exist in ratings
        users = users[users.uid.isin(ratings['uid'])]
        items = items[items.iid.isin(ratings['iid'])]

        users, items, ratings = reindex_df(users, items, ratings)

        data = convert_2_data(users, items, ratings, self.emb_dim, self.repr_dim)

        torch.save(self.collate([data]), self.processed_paths[0])

        # if self.splits is not None:
        #     self.gen_save_ds_property(users, items, ratings, self.splits)

    def __repr__(self):
        return '{}-{}'.format(self.__class__.__name__, self.name.capitalize())


if __name__ == '__main__':
    import os.path as osp
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np

    emb_dim = 300
    repr_dim = 64
    batch_size = 128

    root = osp.join('.', 'tmp', 'ml')
    dataset = MovieLens(root, '1m')
    data = dataset.data
    edge_iter = DataLoader(TensorDataset(data.edge_index.t(), data.edge_attr), batch_size=batch_size, shuffle=True)

    loss_func = torch.nn.MSELoss()
    opt = torch.optim.SGD([data.x, data.r_proj, data.r_emb], lr=1e-3)

    for i in range(20):
        losses = []
        pbar = tqdm.tqdm(edge_iter, total=len(edge_iter))
        for batch in pbar:
            edge_index, edge_attr = batch
            r_idx = edge_attr[:, 0]
            x = data.x
            r_emb = data.r_emb[r_idx]
            r_proj = data.r_proj[r_idx].reshape(-1, emb_dim, repr_dim)
            proj_head = torch.matmul(x[edge_index[:, :1]], r_proj).reshape(-1, repr_dim)
            proj_tail = torch.matmul(x[edge_index[:, 1:2]], r_proj).reshape(-1, repr_dim)

            loss = loss_func(r_emb + proj_head, proj_tail.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.detach()))
            pbar.set_description('loss: {}'.format(np.mean(losses)))



    # head_emb = data.x[data.edge_idx[:, :1]]
    # tail_emb = data.x[data.edge_idx[:, 1:2]]
    # relation = data.r_emb[data.edge_attr[:, :1]]
    # proj_head_emb = torch.mul()
    #
    # r_proj = data.r_proj[data.edge_attr[:, :1]]
    # r_proj = r_proj.reshape(r_proj.shape[0], emb_dim, repr_dim)
    #
    # opt = torch.optim.SGD()
    # loss_func = torch.nn.MSE()
    #
    # for i in range(10):
    #     loss =