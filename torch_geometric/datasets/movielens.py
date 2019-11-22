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
    iid2raw_iid = {iid: raw_iid for raw_iid, iid in zip(raw_iids, iids)}

    rating_uids = np.array(interactions.uid, dtype=np.int)
    rating_iids = np.array(interactions.iid, dtype=np.int)
    print('reindex user id of ratings...')
    rating_uids = [raw_uid2uid[rating_uid] for rating_uid in rating_uids]
    print('reindex item id of ratings...')
    rating_iids = [raw_iid2iid[rating_iid] for rating_iid in rating_iids]
    interactions.loc[:, 'uid'] = rating_uids
    interactions.loc[:, 'iid'] = rating_iids

    return users, items, interactions, iid2raw_iid


def convert_2_data(
        users, items, ratings, iid2raw_iid,
        train_ratio, sec_order):
    """
    Entitiy node include (gender, occupation, genres)

    n_nodes = n_users + n_items + n_genders + n_occupation + n_ages + n_genres

    """
    n_users = users.shape[0]
    n_items = items.shape[0]

    genders = ['M', 'F']
    n_genders = len(genders)

    occupations = list(users.occupation.unique())
    n_occupations = len(occupations)

    ages = ['1', '18', '25', '35', '45', '50', '56']
    n_ages = len(ages)

    genres = list(items.keys()[3:21])
    n_genres = len(genres)

    # Bulid node id
    num_nodes = n_users + n_items + n_genders + n_occupations + n_ages + n_genres

    # Build property2id map
    users['node_id'] = users['uid']
    items['node_id'] = items['iid'] + n_users
    user_node_id_map = {uid: i for i, uid in enumerate(users['uid'].values)}
    item_node_id_map = {iid: n_users + i for i, iid in enumerate(items['iid'].values)}
    gender_node_id_map = {gender: n_users + n_items + i for i, gender in enumerate(genders)}
    occupation_node_id_map = {occupation: n_users + n_items + n_genders + i for i, occupation in enumerate(occupations)}
    genre_node_id_map = {genre: n_users + n_items + n_genders + n_occupations + i for i, genre in enumerate(genres)}
    age_node_id_map = {genre: n_users + n_items + n_genders + n_occupations + n_ages + i for i, genre in enumerate(ages)}

    # Start creating edges
    row_idx, col_idx = [], []
    edge_attrs = []

    rating_begin = 0

    print('Creating user property edges...')
    for _, row in tqdm.tqdm(users.iterrows(), total=users.shape[0]):
        gender = row['gender']
        occupation = row['occupation']
        age = row['age']

        u_nid = row['uid']
        gender_nid = gender_node_id_map[gender]
        row_idx.append(u_nid)
        col_idx.append(gender_nid)

        occupation_nid = occupation_node_id_map[occupation]
        row_idx.append(u_nid)
        col_idx.append(occupation_nid)

        age_nid = age_node_id_map[age]
        row_idx.append(u_nid)
        col_idx.append(age_nid)
    edge_attrs += [-1 for i in range(3 * users.shape[0])]
    rating_begin += 3 * users.shape[0]

    print('Creating item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        i_nid = item_node_id_map[row['iid']]

        for genre in genres:
            if not row[genre]:
                continue
            g_nid = genre_node_id_map[genre]
            row_idx.append(i_nid)
            col_idx.append(g_nid)
            edge_attrs.append(-1)
            rating_begin += 1

    print('Creating rating property edges...')
    row_idx += list(users.iloc[ratings['uid']]['node_id'].values)
    col_idx += list(items.iloc[ratings['iid']]['node_id'].values)
    edge_attrs += list(ratings['rating'])

    print('Building masks...')
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
        age = row['age']

        u_nid = row['uid']
        gender_nid = gender_node_id_map[gender]
        col_idx.append(u_nid)
        row_idx.append(gender_nid)

        occupation_nid = occupation_node_id_map[occupation]
        col_idx.append(u_nid)
        row_idx.append(occupation_nid)

        age_nid = age_node_id_map[age]
        col_idx.append(u_nid)
        row_idx.append(age_nid)
    edge_attrs += [-1 for i in range(3 * users.shape[0])]

    print('Creating reverse item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        i_nid = item_node_id_map[row['iid']]

        for genre in genres:
            if not row[genre]:
                continue
            g_nid = genre_node_id_map[genre]
            col_idx.append(i_nid)
            row_idx.append(g_nid)
            edge_attrs.append(-1)

    print('Creating reverse rating property edges...')
    col_idx += list(users.iloc[ratings['uid']]['node_id'].values)
    row_idx += list(items.iloc[ratings['iid']]['node_id'].values)
    edge_attrs += list(ratings['rating'])

    row_idx = [int(idx) for idx in row_idx]
    col_idx = [int(idx) for idx in col_idx]
    row_idx = np.array(row_idx).reshape(1, -1)
    col_idx = np.array(col_idx).reshape(1, -1)
    edge_index = np.concatenate((row_idx, col_idx), axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    edge_attrs = np.array(edge_attrs)
    edge_attrs = torch.from_numpy(edge_attrs).long().t()

    kwargs = {
        'num_nodes': num_nodes,
        'edge_index': edge_index, 'edge_attr': edge_attrs,
        'rating_edge_mask': rating_edge_mask,
        'users': users, 'ratings': ratings, 'items': items,
        'iid2raw_iid': iid2raw_iid,
        'user_node_id_map': user_node_id_map,
        'gender_node_id_map': gender_node_id_map, 'occupation_node_id_map': occupation_node_id_map,
        'age_node_id_map': age_node_id_map, 'genre_node_id_map': genre_node_id_map
    }

    if train_ratio is not None:
        kwargs['train_edge_mask'] = train_edge_mask
        kwargs['test_edge_mask'] = test_edge_mask
        if sec_order:
            print('Creating second order edges...')
            kwargs['train_sec_order_edge_index'] = \
                get_sec_order_edge(edge_index[:, train_edge_mask])
            kwargs['n_sec_order_edge'] = kwargs['train_sec_order_edge_index'].shape[1]
    else:
        if sec_order:
            print('Creating second order edges...')
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
                 n_core=10,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.name = name.lower()
        assert self.name in ['1m']
        self.n_core = n_core
        self.sec_order = sec_order

        self.train_ratio = kwargs.get('train_ratio', None)
        self.debug = kwargs.get('debug', False)
        self.seed = kwargs.get('seed', None)
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

        users, items, ratings, iid2raw_iid = reindex_df(users, items, ratings)

        data = convert_2_data(users, items, ratings, iid2raw_iid, self.train_ratio, self.sec_order)

        torch.save(self.collate([data]), self.processed_paths[0], pickle_protocol=4)

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
        if self.seed is not None:
            suffixes.append('seed_{}'.format(self.seed))
        if not suffixes:
            suffix = ''
        else:
            suffix = '_'.join(suffixes)
        return '_' + suffix

