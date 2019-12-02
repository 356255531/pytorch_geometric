import torch
from os.path import join
from os.path import isfile
import numpy as np
import random as rd
import tqdm
import itertools
from collections import Counter

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_ml
from torch_geometric.data import Data, extract_zip
from torch_geometric.utils import get_sec_order_edge


def save_df(df, path):
    df.to_csv(path, encoding='utf-8')


def reindex_df(users, items, interactions):
    """
    reindex users, items, interactions in case there are some values missing in between
    :param users: pd.DataFrame
    :param items: pd.DataFrame
    :param interactions: pd.DataFrame
    :return: same
    """
    num_users = users.shape[0]
    num_movies = items.shape[0]

    raw_uids = np.array(users.uid, dtype=np.int)
    raw_iids = np.array(items.iid, dtype=np.int)
    uids = np.arange(num_users)
    iids = np.arange(num_movies)

    users.loc[:, 'uid'] = uids
    items.loc[:, 'iid'] = iids

    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(raw_uids, uids)}
    raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(raw_iids, iids)}

    rating_uids = np.array(interactions.uid, dtype=np.int)
    rating_iids = np.array(interactions.iid, dtype=np.int)
    print('reindex user id of ratings...')
    rating_uids = [raw_uid2uid[rating_uid] for rating_uid in rating_uids]
    print('reindex item id of ratings...')
    rating_iids = [raw_iid2iid[rating_iid] for rating_iid in rating_iids]
    interactions.loc[:, 'uid'] = rating_uids
    interactions.loc[:, 'iid'] = rating_iids

    return users, items, interactions


def convert_2_data(
        users, items, ratings,
        train_ratio, sec_order):
    """
    Entitiy node include (gender, occupation, genres)
    num_nodes = num_users + num_items + num_genders + num_occupation + num_ages + num_genres + num_years + num_directors + num_actors + num_writers
    """
    num_users = users.shape[0]
    num_items = items.shape[0]

    relations = [
        'gender', 'occupation', 'age', 'genre', 'year', 'director', 'actor', 'writer', 'interact',
        '-gender', '-occupation', '-age', '-genre', '-year', '-director', '-actor', '-writer', '-interact'
    ]

    genders = ['M', 'F']
    num_genders = len(genders)

    occupations = list(users.occupation.unique())
    num_occupations = len(occupations)

    ages = list(users.age.unique())
    num_ages = len(ages)

    genres = list(items.keys()[3:21])
    num_genres = len(genres)

    years = list(np.unique(items.discretized_year.to_numpy()))
    num_years = len(years)

    directors = list(items.director.unique())
    num_directors = len(directors)

    actors = [actor_str.split(', ') for actor_str in items.actor.values]
    actors = list(set(itertools.chain.from_iterable(actors)))
    actors = [actor for actor in actors if actor != '']
    num_actors = len(actors)

    writers = list(items.writer.unique())
    num_writers = len(writers)
    # Bulid node id
    num_nodes = num_users + num_items + num_genders + num_occupations + num_ages + num_genres + num_years + num_directors + num_actors + num_writers

    # Add node id features to users and items
    users['nid'] = users['uid']
    items['nid'] = items['iid'] + num_users

    # Build property2id map
    acc = 0
    user_node_id_map = {uid: i + acc for i, uid in enumerate(users['uid'].values)}
    acc += users.shape[0]
    item_node_id_map = {iid: i + acc for i, iid in enumerate(items['iid'].values)}
    acc += items.shape[0]
    gender_node_id_map = {gender: i + acc for i, gender in enumerate(genders)}
    acc += num_genders
    occupation_node_id_map = {occupation: i + acc for i, occupation in enumerate(occupations)}
    acc += num_occupations
    age_node_id_map = {age: i + acc for i, age in enumerate(ages)}
    acc += num_ages
    genre_node_id_map = {genre: i + acc for i, genre in enumerate(genres)}
    acc += num_genres
    year_node_id_map = {year: i + acc for i, year in enumerate(years)}
    acc += num_years
    director_node_id_map = {director: i + acc for i, director in enumerate(directors)}
    acc += num_directors
    actor_node_id_map = {actor: i + acc for i, actor in enumerate(actors)}
    acc += num_actors
    writer_node_id_map = {writer: i + acc for i, writer in enumerate(writers)}

    # Start creating edges
    row_idx, col_idx = [], []
    edge_attrs = []

    rating_begin = 0

    print('Creating user property edges...')
    for _, row in tqdm.tqdm(users.iterrows(), total=users.shape[0]):
        u_nid = row['nid']
        gender = row['gender']
        occupation = row['occupation']
        age = row['age']

        gender_nid = gender_node_id_map[gender]
        row_idx.append(u_nid)
        col_idx.append(gender_nid)
        edge_attrs.append([relations.index('gender'), -1])

        occupation_nid = occupation_node_id_map[occupation]
        row_idx.append(u_nid)
        col_idx.append(occupation_nid)
        edge_attrs.append([relations.index('occupation'), -1])

        age_nid = age_node_id_map[age]
        row_idx.append(u_nid)
        col_idx.append(age_nid)
        edge_attrs.append([relations.index('age'), -1])
    rating_begin += 3 * users.shape[0]

    print('Creating item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        i_nid = row['nid']
        year = row['discretized_year']
        director = row['director']
        actors = row['actor'].split(', ')
        writer = row['writer']

        y_nid = year_node_id_map[year]
        row_idx.append(i_nid)
        col_idx.append(y_nid)
        edge_attrs.append([relations.index('year'), -1])

        if director != '':
            d_nid = director_node_id_map[director]
            row_idx.append(i_nid)
            col_idx.append(d_nid)
            edge_attrs.append([relations.index('director'), -1])
            rating_begin += 1

        for actor in actors:
            if actor != '':
                a_nid = actor_node_id_map[actor]
                row_idx.append(i_nid)
                col_idx.append(a_nid)
                edge_attrs.append([relations.index('actor'), -1])
                rating_begin += 1

        if writer != '':
            w_nid = writer_node_id_map[writer]
            row_idx.append(i_nid)
            col_idx.append(w_nid)
            edge_attrs.append([relations.index('writer'), -1])
            rating_begin += 1

        for genre in genres:
            if not row[genre]:
                continue
            g_nid = genre_node_id_map[genre]
            row_idx.append(i_nid)
            col_idx.append(g_nid)
            edge_attrs.append([relations.index('genre'), -1])
            rating_begin += 1
    rating_begin += items.shape[0]

    print('Creating rating property edges...')
    row_idx += list(users.iloc[ratings['uid']]['nid'].values)
    col_idx += list(items.iloc[ratings['iid']]['nid'].values)
    edge_attrs += [[relations.index('interact'), i] for i in list(ratings['rating'])]

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
        test_rating_idx = list(set(range(ratings.shape[0])) - set(train_rating_idx))
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
        u_nid = row['uid']
        gender = row['gender']
        occupation = row['occupation']
        age = row['age']

        gender_nid = gender_node_id_map[gender]
        col_idx.append(u_nid)
        row_idx.append(gender_nid)
        edge_attrs.append([relations.index('-gender'), -1])

        occupation_nid = occupation_node_id_map[occupation]
        col_idx.append(u_nid)
        row_idx.append(occupation_nid)
        edge_attrs.append([relations.index('-occupation'), -1])

        age_nid = age_node_id_map[age]
        col_idx.append(u_nid)
        row_idx.append(age_nid)
        edge_attrs.append([relations.index('-age'), -1])

    print('Creating reverse item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        i_nid = row['nid']
        year = row['discretized_year']
        director = row['director']
        actors = row['actor'].split(', ')
        writer = row['writer']

        y_nid = year_node_id_map[year]
        col_idx.append(i_nid)
        row_idx.append(y_nid)
        edge_attrs.append([relations.index('-year'), -1])

        if director != '':
            d_nid = director_node_id_map[director]
            col_idx.append(i_nid)
            row_idx.append(d_nid)
            edge_attrs.append([relations.index('-director'), -1])

        for actor in actors:
            if actor != '':
                a_nid = actor_node_id_map[actor]
                col_idx.append(i_nid)
                row_idx.append(a_nid)
                edge_attrs.append([relations.index('-actor'), -1])

        if writer != '':
            w_nid = writer_node_id_map[writer]
            col_idx.append(i_nid)
            row_idx.append(w_nid)
            edge_attrs.append([relations.index('-writer'), -1])
            rating_begin += 1

        for genre in genres:
            if not row[genre]:
                continue
            g_nid = genre_node_id_map[genre]
            col_idx.append(i_nid)
            row_idx.append(g_nid)
            edge_attrs.append([relations.index('-genre'), -1])

    print('Creating reverse rating property edges...')
    col_idx += list(users.iloc[ratings['uid']]['nid'].values)
    row_idx += list(items.iloc[ratings['iid']]['nid'].values)
    edge_attrs += [[relations.index('-interact'), i] for i in list(ratings['rating'])]

    row_idx = [int(idx) for idx in row_idx]
    col_idx = [int(idx) for idx in col_idx]
    row_idx = np.array(row_idx).reshape(1, -1)
    col_idx = np.array(col_idx).reshape(1, -1)
    edge_index = np.concatenate((row_idx, col_idx), axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    edge_attrs = np.array(edge_attrs)
    edge_attrs = torch.from_numpy(edge_attrs).long()

    kwargs = {
        'num_nodes': num_nodes,
        'edge_index': edge_index, 'edge_attr': edge_attrs,
        'rating_edge_mask': rating_edge_mask,
        'users': users, 'ratings': ratings, 'items': items,
        'relations': relations, 'num_relations': len(relations),
        'user_node_id_map': user_node_id_map, 'gender_node_id_map': gender_node_id_map,
        'occupation_node_id_map': occupation_node_id_map, 'age_node_id_map': age_node_id_map,
        'genre_node_id_map': genre_node_id_map, 'year_node_id_map': year_node_id_map,
        'actor_node_id_map': actor_node_id_map, 'director_node_id_map': director_node_id_map,
        'writer_node_id_map': writer_node_id_map
    }

    if train_ratio is not None:
        kwargs['train_rating_idx'] = train_rating_idx
        kwargs['test_rating_idx'] = test_rating_idx
        kwargs['train_edge_mask'] = train_edge_mask
        kwargs['test_edge_mask'] = test_edge_mask
        if sec_order:
            print('Creating second order edges...')
            kwargs['train_sec_order_edge_index'] = \
                get_sec_order_edge(edge_index[:, train_edge_mask])
            kwargs['num_sec_order_edge'] = kwargs['train_sec_order_edge_index'].shape[1]
    else:
        if sec_order:
            print('Creating second order edges...')
            kwargs['sec_order_edge_index'] = get_sec_order_edge(edge_index)
            kwargs['num_sec_order_edge'] = kwargs['sec_order_edge_index'].shape[1]

    return Data(**kwargs)


class MovieLens(InMemoryDataset):
    url = 'http://files.grouplens.org/datasets/movielens/'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.name = name.lower()
        assert self.name in ['1m']
        self.num_core = kwargs.get('num_core', 10)
        self.sec_order = kwargs.get('sec_order', False)
        self.implicit = kwargs.get('implicit', False)
        self.train_ratio = kwargs.get('train_ratio', None)
        self.debug = kwargs.get('debug', False)
        self.seed = kwargs.get('seed', None)
        self.suffix = self.build_suffix()
        super(MovieLens, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.implicit:
            self.data.edge_attr[self.data.rating_edge_mask[0], 1] = 1
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
        if isfile(join(self.processed_dir, 'movies.pkl')) and isfile(join(self.processed_dir, 'ratings.pkl')) and isfile(join(self.processed_dir, 'users.pkl')):
            print('Read data frame!')
            users, items, ratings = read_ml(self.processed_dir, True, self.debug)
            users = users.fillna('')
            items = items.fillna('')
            ratings = ratings.fillna('')
        else:
            print('Read from raw data!')
            users, items, ratings = read_ml(unzip_raw_dir, False, self.debug)

            # Discretized year
            years = items.year.to_numpy()
            min_year = min(years)
            max_year = max(years)
            num_years = (max_year - min_year) // 10
            discretized_years = [min_year + i * 10 for i in range(num_years + 1)]
            for i, year in enumerate(discretized_years):
                if i == 0:
                    years[years <= year] = year
                else:
                    years[(years <= year) & (years > discretized_years[i - 1])] = year
            items['discretized_year'] = years

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

            # delete ratings have movie and user count less than self.num_core
            ratings = ratings[ratings.movie_count > self.num_core]
            ratings = ratings[ratings.user_count > self.num_core]

            # drop users and movies which do not exist in ratings
            users = users[users.uid.isin(ratings['uid'])]
            items = items[items.iid.isin(ratings['iid'])]

            # Drop the unfrequent writer, actor and directors
            writers = items.writer.values
            writers_dict = Counter(writers)
            unique_writers = {k: v for k, v in writers_dict.items() if v > self.num_core}.keys()
            writers = [writer if writer in unique_writers else '' for writer in writers]
            directors = items.director.values
            directors_dict = Counter(directors)
            unique_directors = {k: v for k, v in directors_dict.items() if v > self.num_core}.keys()
            directors = [director if director in unique_directors else '' for director in directors]
            actor_strs = [actor_str for actor_str in items.actor.values]
            actors = [actor_str.split(', ') for actor_str in actor_strs]
            actors = list(itertools.chain.from_iterable(actors))
            actors_dict = Counter(actors)
            unique_actors = {k: v for k, v in actors_dict.items() if v > self.num_core}.keys()
            actor_strs = [[single_actor_str for single_actor_str in actor_str.split(', ') if single_actor_str in unique_actors] for actor_str in actor_strs]
            actor_strs = [', '.join(actor_str) for actor_str in actor_strs]
            items['writer'] = writers
            items['director'] = directors
            items['actor'] = actor_strs

            users, items, ratings = reindex_df(users, items, ratings)
            save_df(users, join(self.processed_dir, 'users.pkl'))
            save_df(items, join(self.processed_dir, 'movies.pkl'))
            save_df(ratings, join(self.processed_dir, 'ratings.pkl'))

        data = convert_2_data(users, items, ratings, self.train_ratio, self.sec_order)

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
