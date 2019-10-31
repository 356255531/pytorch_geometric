import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_ml
from torch_geometric.data import Data, extract_zip
from torch_geometric.utils import get_sec_order_edge

from os.path import join
import numpy as np
import random as rd
import tqdm
import pickle
import pandas


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
        train_ratio, sec_order, tensor_type):
    """
    Entitiy node include (gender, occupation, genres)

    n_nodes = n_users + n_items + n_genders + n_occupation + n_genres

    """
    float_tensor, int_tensor, bool_tensor = tensor_type

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

    x = torch.nn.Embedding(n_nodes, emb_dim, max_norm=1, norm_type=2.0).type(float_tensor).weight

    r_emb = torch.nn.Embedding(n_relations, repr_dim, max_norm=1, norm_type=2.0).type(float_tensor).weight
    r_proj = torch.nn.Embedding(n_relations // 2, emb_dim * repr_dim, max_norm=1, norm_type=2.0).type(float_tensor).weight
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
        i_nid = n_users + row['iid']

        row_idx.append(u_nid)
        col_idx.append(i_nid)
        edge_attrs.append([relation_map['interact'], row['rating']])

    rating_mask = torch.ones(ratings.shape[0])
    rating_edge_mask = torch.cat(
        (
            torch.zeros(rating_begin),
            rating_mask,
            torch.zeros(rating_begin),
            rating_mask),
    ).type(bool_tensor)
    if train_ratio is not None:
        train_rating_mask = torch.zeros(ratings.shape[0])
        test_rating_mask = torch.ones(ratings.shape[0])
        train_rating_idx = rd.sample([i for i in range(ratings.shape[0])], int(ratings.shape[0]*0.8))
        train_rating_mask[train_rating_idx] = 1
        test_rating_mask[train_rating_idx] = 0

        train_edge_mask = torch.cat(
            (
                torch.ones(rating_begin),
                train_rating_mask,
                torch.ones(rating_begin),
                train_rating_mask)
        ).type(bool_tensor)

        test_edge_mask = torch.cat(
            (
                torch.ones(rating_begin),
                test_rating_mask,
                torch.ones(rating_begin),
                test_rating_mask)
        ).type(bool_tensor)

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
        i_nid = n_users + row['iid']

        col_idx.append(u_nid)
        row_idx.append(i_nid)
        edge_attrs.append([relation_map['-interact'], row['rating']])

    row_idx = np.array(row_idx).reshape(1, -1)
    col_idx = np.array(col_idx).reshape(1, -1)
    edge_index = np.concatenate((row_idx, col_idx), axis=0)
    edge_index = torch.from_numpy(edge_index).type(int_tensor)
    edge_attrs = np.array(edge_attrs)
    edge_attrs = torch.from_numpy(edge_attrs).type(int_tensor)

    kwargs = {
        'x': x, 'edge_index': edge_index, 'edge_attr': edge_attrs,
        'rating_edge_mask': rating_edge_mask, 'r_emb': r_emb, 'r_proj': r_proj,
        'n_users': n_users, 'n_items': n_items, 'n_interactions': n_interactions,
        'relation_map': relation_map

    }
    if train_ratio is not None:
        kwargs['train_edge_mask'] = train_edge_mask
        kwargs['test_edge_mask'] = test_edge_mask
        if sec_order:
            kwargs['train_sec_order_edge_index'] = \
                get_sec_order_edge(edge_index[:, train_edge_mask], x).type(int_tensor)
            kwargs['test_sec_order_edge_index'] = \
                get_sec_order_edge(edge_index, x).type(int_tensor)
    else:
        kwargs['sec_order_edge_index'] = get_sec_order_edge(edge_index, x).type(int_tensor)

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
                 emb_dim=300,
                 repr_dim=64,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.name = name.lower()
        assert self.name in ['1m']
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim

        self.train_ratio = kwargs.get('train_ratio', None)
        self.debug = kwargs.get('debug', False)
        self.suffix = self.build_suffix()

        self.sec_order = kwargs.get('sec_order', None)
        if kwargs.get('tensor_type', None) is None:
            float_type = torch.float16
            int_type = torch.int16
            bool_type = torch.bool
            self.tensor_type = (float_type, int_type, bool_type)
        else:
            self.tensor_type = kwargs.get('tensor_type', None)
        super(MovieLens, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

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

        # delete ratings have movie and user count less than 10
        ratings = ratings[ratings.movie_count > 9]
        ratings = ratings[ratings.user_count > 9]

        # drop users and movies which do not exist in ratings
        users = users[users.uid.isin(ratings['uid'])]
        items = items[items.iid.isin(ratings['iid'])]

        users, items, ratings = reindex_df(users, items, ratings)

        data = convert_2_data(
            users, items, ratings,
            self.emb_dim, self.repr_dim, self.train_ratio, self.sec_order,
            tensor_type=self.tensor_type)

        torch.save(self.collate([data]), self.processed_paths[0])

    def build_suffix(self):
        suffixes = []
        if self.train_ratio is not None:
            suffixes.append('train')
        if self.debug:
            suffixes.append('debug')
        if not suffixes:
            suffix = ''
        else:
            suffix = '_'.join(suffixes)
        return '_' + suffix

    def __repr__(self):
        return '{}-{}'.format(self.__class__.__name__, self.name.capitalize())