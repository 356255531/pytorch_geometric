import torch
from os.path import join
import numpy as np
import random as rd
import tqdm
import pickle


from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_ml
from torch_geometric.data import Data, extract_zip
from torch_geometric.utils import get_sec_order_edge


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


def convert_2_data(
        users, items, ratings,
        emb_dim, repr_dim,
        train_ratio, sec_order):
    """
    Convert pandas.DataFrame to torch-geometirc graph dataset

    Node include user nodes, item nodes and entity nodes (gender, occupation, genres)
    Number and the order of nodes see below:
        n_nodes = n_users + n_items + n_genders + n_occupation + n_genres

    """
    n_users = users.shape[0]
    n_items = items.shape[0]

    genders = ['M', 'F']
    n_genders = len(genders)

    occupations = list(users.occupation.unique())
    n_occupations = len(occupations)

    genres = list(items.keys()[3:21])
    n_genres = len(genres)

    relations = ['gender', 'occupation', 'genre', 'interact', '-gender', '-occupation', '-genre', '-interact']
    n_relations = len(relations)

    # Bulid node id
    n_nodes = n_users + n_items + n_genders + n_occupations + n_genres
    users['node_id'] = users['uid']
    items['node_id'] = n_users + items['iid']
    genders_node_id_map = {gender: n_users + n_items + i for i, gender in enumerate(genders)}
    occupation_node_id_map = {occupation: n_users + n_items + n_genders + i for i, occupation in enumerate(occupations)}
    genres_node_id_map = {genre: n_users + n_items + n_genders + n_occupations + i for i, genre in enumerate(genres)}

    # Node embedding
    x = torch.nn.Embedding(n_nodes, emb_dim, max_norm=1, norm_type=2.0).weight

    # Relation embedding
    r_emb = torch.nn.Embedding(n_relations, repr_dim, max_norm=1, norm_type=2.0).weight
    # Knowledge graph projection embedding
    r_proj = torch.nn.Embedding(
        n_relations // 2, emb_dim * repr_dim, max_norm=1, norm_type=2.0
    ).weight
    r_proj = torch.cat((r_proj, -r_proj), dim=0)

    # Start creating edges
    row_idx, col_idx = [], []
    edge_attrs = []

    rating_begin = 0

    print('Creating user property edges...')
    for _, row in tqdm.tqdm(users.iterrows(), total=users.shape[0]):
        gender = row['gender']
        occupation = row['occupation']

        u_nid = row['uid']
        gender_nid = genders_node_id_map[gender]
        row_idx.append(u_nid)
        col_idx.append(gender_nid)
        edge_attrs.append([relations.index('gender'), -1])

        occupation_nid = occupation_node_id_map[occupation]
        row_idx.append(u_nid)
        col_idx.append(occupation_nid)
        edge_attrs.append([relations.index('occupation'), -1])
    rating_begin += 2 * users.shape[0]

    print('Creating item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        i_nid = row['node_id']

        for genre in genres:
            if not row[genre]:
                continue
            g_nid = genres_node_id_map[genre]
            row_idx.append(i_nid)
            col_idx.append(g_nid)
            edge_attrs.append([relations.index('genre'), -1])
            rating_begin += 1

    print('Creating rating property edges...')
    for _, row in tqdm.tqdm(ratings.iterrows(), total=ratings.shape[0]):
        u_nid = users.loc[users['uid'] == row['uid']]['node_id']
        i_nid = items.loc[items['iid'] == row['iid']]['node_id']

        row_idx.append(u_nid)
        col_idx.append(i_nid)
        edge_attrs.append([relations.index('interact'), row['rating']])

    # Create masks
    rating_mask = torch.ones(ratings.shape[0], dtype=torch.bool)
    rating_edge_mask = torch.cat(
        (
            torch.zeros(rating_begin, dtype=torch.bool),
            rating_mask,
            torch.zeros(rating_begin, dtype=torch.bool),
            rating_mask),
    )
    if train_ratio is not None:
        train_rating_mask = torch.zeros(ratings.shape[0], dtype=torch.bool)
        test_rating_mask = torch.ones(ratings.shape[0], dtype=torch.bool)
        train_rating_idx = rd.sample([i for i in range(ratings.shape[0])], int(ratings.shape[0] * train_ratio))
        train_rating_mask[train_rating_idx] = 1
        test_rating_mask[train_rating_idx] = 0

        train_edge_mask = torch.cat(
            (
                torch.ones(rating_begin, dtype=torch.bool),
                train_rating_mask,
                torch.ones(rating_begin, dtype=torch.bool),
                train_rating_mask)
        )

        test_edge_mask = torch.cat(
            (
                torch.ones(rating_begin, dtype=torch.bool),
                test_rating_mask,
                torch.ones(rating_begin, dtype=torch.bool),
                test_rating_mask)
        )

    print('Creating reverse user property edges...')
    for _, row in tqdm.tqdm(users.iterrows(), total=users.shape[0]):
        gender = row['gender']
        occupation = row['occupation']

        u_nid = row['uid']
        gender_nid = genders_node_id_map[gender]
        col_idx.append(u_nid)
        row_idx.append(gender_nid)
        edge_attrs.append([relations.index('-gender'), -1])

        occupation_nid = occupation_node_id_map[occupation]
        col_idx.append(u_nid)
        row_idx.append(occupation_nid)
        edge_attrs.append([relations.index('-occupation'), -1])
    rating_begin += 2 * users.shape[0]

    print('Creating reverse item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        i_nid = row['node_id']

        for genre in genres:
            if not row[genre]:
                continue
            g_nid = genres_node_id_map[genre]
            col_idx.append(i_nid)
            row_idx.append(g_nid)
            edge_attrs.append([relations.index('-genre'), -1])
            rating_begin += 1

    print('Creating reverse rating property edges...')
    for _, row in tqdm.tqdm(ratings.iterrows(), total=ratings.shape[0]):
        u_nid = users.loc[users['uid'] == row['uid']]['node_id']
        i_nid = items.loc[items['iid'] == row['iid']]['node_id']

        col_idx.append(u_nid)
        row_idx.append(i_nid)
        edge_attrs.append([relations.index('-interact'), row['rating']])

    row_idx = [int(idx) for idx in row_idx]
    col_idx = [int(idx) for idx in col_idx]
    row_idx = np.array(row_idx).reshape(1, -1)
    col_idx = np.array(col_idx).reshape(1, -1)
    edge_index = np.concatenate((row_idx, col_idx), axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    edge_attrs = np.array(edge_attrs)
    edge_attrs = torch.from_numpy(edge_attrs)

    kwargs = {
        'x': x, 'edge_index': edge_index, 'edge_attr': edge_attrs,
        'rating_edge_mask': rating_edge_mask, 'r_emb': r_emb, 'r_proj': r_proj,
        'users': users, 'ratings': ratings, 'movies': items,
        'genders_node_id_map': genders_node_id_map, 'occupation_node_id_map': occupation_node_id_map,
        'genres_node_id_map': genres_node_id_map
    }

    print('Creating second order edges...')
    if train_ratio is not None:
        kwargs['train_edge_mask'] = train_edge_mask
        kwargs['test_edge_mask'] = test_edge_mask
        if sec_order:
            kwargs['train_sec_order_edge_index'] = \
                get_sec_order_edge(edge_index[:, train_edge_mask])
            kwargs['n_sec_order_edge'] = kwargs['train_sec_order_edge_index'].shape[1]
    else:
        kwargs['sec_order_edge_index'] = get_sec_order_edge(edge_index)
        kwargs['n_sec_order_edge'] = kwargs['sec_order_edge_index'].shape[1]

    return Data(**kwargs)


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
                 sec_order=False,
                 emb_dim=300,
                 repr_dim=64,
                 n_core=10,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.name = name.lower()
        assert self.name in ['1m']
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim
        self.n_core = n_core

        self.train_ratio = kwargs.get('train_ratio', None)
        self.debug = kwargs.get('debug', False)
        self.sec_order = sec_order
        self.suffix = self.build_suffix()
        super(MovieLens, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        print('Graph params: {}'.format(self.data))

        print('Dataset loaded!')

    @property
    def raw_file_names(self):
        return 'ml-{}.zip'.format(self.name.lower())

    @property
    def processed_file_names(self):
        return ['data{}.pt'.format(self.suffix)]

    def download(self):
        path = download_url(self.url + self.raw_file_names, self.raw_dir)

        extract_zip(path, self.raw_dir)

    def process(self):
        unzip_raw_dir = join(self.raw_dir, 'ml-{}'.format(self.name))

        # read files
        users, items, ratings = read_ml(unzip_raw_dir, self.debug)

        # remove duplications
        users = users.drop_duplicates()
        items = items.drop_duplicates()
        ratings = ratings.drop_duplicates()

        item_count = ratings['iid'].value_counts()
        user_count = ratings['uid'].value_counts()
        item_count.name = 'movie_count'
        user_count.name = 'user_count'
        ratings = ratings.join(item_count, on='iid')
        ratings = ratings.join(user_count, on='uid')

        # delete ratings have movie and user count less than self.n_core
        ratings = ratings[ratings.movie_count > self.n_core]
        ratings = ratings[ratings.user_count > self.n_core]

        # drop users and movies which do not exist in ratings
        users = users[users.uid.isin(ratings['uid'])]
        items = items[items.iid.isin(ratings['iid'])]

        users, items, ratings = reindex_df(users, items, ratings)

        data = convert_2_data(
            users, items, ratings,
            self.emb_dim, self.repr_dim, self.train_ratio, self.sec_order)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}-{}'.format(self.__class__.__name__, self.name.capitalize())

    def build_suffix(self):
        suffixes = []
        if self.train_ratio is not None:
            suffixes.append('train_{}'.format(self.train_ratio))
        if self.debug:
            suffixes.append('debug_{}'.format(self.debug))
        if self.sec_order:
            suffixes.append('sec_order')
        if not suffixes:
            suffix = ''
        else:
            suffix = '_'.join(suffixes)
        return '_' + suffix


if __name__ == '__main__':
    import torch
    from torch_geometric.datasets import MovieLens
    import os.path as osp
    import pdb

    torch.random.manual_seed(2019)

    emb_dim = 300
    repr_dim = 64
    batch_size = 128

    root = osp.join('.', 'tmp', 'ml')
    dataset = MovieLens(root, '1m', debug=0.1, train_ratio=0.8)
    data = dataset.data
    pdb.set_trace()
