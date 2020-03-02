import torch
from os.path import join
from os.path import isfile
import numpy as np
import pandas as pd
import random as rd
import tqdm
import itertools
from collections import Counter

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_ml
from torch_geometric.data import Data, extract_zip
from torch_geometric.utils import create_path, filter_path


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


def create_user_pos_neg_pair(ratings, train_rating_idx, test_rating_mask, e2nid):
    u_nids = e2nid['uid'].values()
    i_nids = e2nid['iid'].values()

    all_pairs = [[u_nid, i_nid] for u_nid, i_nid in itertools.product(u_nids, i_nids)]
    pos_paris = [[e2nid['uid'][uid], e2nid['iid'][iid]] for uid, iid in zip(ratings.uid, ratings.iid)]

    train_pos_pairs = [[e2nid['uid'][uid], e2nid['iid'][iid]] for uid, iid in zip(ratings.iloc[train_rating_idx].uid, ratings.iloc[train_rating_idx].iid)]
    test_pos_pairs = [[e2nid['uid'][uid], e2nid['iid'][iid]] for uid, iid in zip(ratings.iloc[test_rating_mask].uid, ratings.iloc[test_rating_mask].iid)]
    train_pos_pairs_df = pd.DataFrame(data=np.array(train_pos_pairs), columns=['u_nid', 'pos_i_nid'])
    test_pos_pairs_df = pd.DataFrame(data=np.array(test_pos_pairs), columns=['u_nid', 'pos_i_nid'])

    neg_pairs = set([tuple(l) for l in all_pairs]).difference([tuple(l) for l in pos_paris])
    neg_pairs = [list(i) for i in neg_pairs]
    neg_pairs_df = pd.DataFrame(data=np.array(neg_pairs), columns=['u_nid', 'neg_i_nid'])

    train_user_pos_neg_pair = pd.merge(train_pos_pairs_df, neg_pairs_df, on='u_nid', how='inner').to_numpy()
    test_user_pos_neg_pair = pd.merge(test_pos_pairs_df, neg_pairs_df, on='u_nid', how='inner').to_numpy()

    return train_user_pos_neg_pair, test_user_pos_neg_pair


def convert_2_data(
        users, items, ratings,
        train_ratio, step_length,
        directed=True
):
    """
    Entitiy node include (gender, occupation, genres)
    num_nodes = num_users + num_items + num_genders + num_occupation + num_ages + num_genres + num_years + num_directors + num_actors + num_writers
    """
    num_users = users.shape[0]
    num_items = items.shape[0]

    #########################  Define relationship  #########################
    if directed:
        relations = [
            'gender', 'occupation', 'age', 'genre', 'year', 'director', 'actor', 'writer', 'interact',
            '-interact'
        ]
    else:
        relations = [
            'gender', 'occupation', 'age', 'genre', 'year', 'director', 'actor', 'writer', 'interact',
            '-gender', '-occupation', '-age', '-genre', '-year', '-director', '-actor', '-writer', '-interact'
        ]

    relation2index = {r: i for i, r in enumerate(relations)}
    index2relation = {i: r for r, i in relation2index.items()}

    #########################  Define entities  #########################
    genders = ['M', 'F']
    num_genders = len(genders)

    occupations = list(users.occupation.unique())
    num_occupations = len(occupations)

    ages = list(users.age.unique())
    num_ages = len(ages)

    genres = list(items.keys()[4:21])
    print(genres)
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
    writers = [writer for writer in writers if writer != '']
    num_writers = len(writers)

    #########################  Define number of entities  #########################
    num_nodes = num_users + num_items + num_genders + num_occupations + num_ages + num_genres + num_years + \
                num_directors + num_actors + num_writers

    #########################  Define entities to node id map  #########################
    acc = 0
    uid2nid = {uid: i + acc for i, uid in enumerate(users['uid'].values)}
    nid2e = {i + acc: ('uid', uid) for i, uid in enumerate(users['uid'].values)}
    acc += users.shape[0]
    iid2nid = {iid: i + acc for i, iid in enumerate(items['iid'].values)}
    for i, iid in enumerate(items['iid'].values):
        nid2e[i + acc] = ('iid', iid)
    acc += items.shape[0]
    gender2nid = {gender: i + acc for i, gender in enumerate(genders)}
    for i, gender in enumerate(genders):
        nid2e[i + acc] = ('gender', gender)
    acc += num_genders
    occ2nid = {occupation: i + acc for i, occupation in enumerate(occupations)}
    for i, occ in enumerate(occupations):
        nid2e[i + acc] = ('occ', occ)
    acc += num_occupations
    age2nid = {age: i + acc for i, age in enumerate(ages)}
    for i, age in enumerate(ages):
        nid2e[i + acc] = ('age', age)
    acc += num_ages
    genre2nid = {genre: i + acc for i, genre in enumerate(genres)}
    for i, genre in enumerate(genres):
        nid2e[i + acc] = ('genre', genre)
    acc += num_genres
    year2nid = {year: i + acc for i, year in enumerate(years)}
    for i, year in enumerate(years):
        nid2e[i + acc] = ('year', year)
    acc += num_years
    director2nid = {director: i + acc for i, director in enumerate(directors)}
    for i, director in enumerate(directors):
        nid2e[i + acc] = ('director', director)
    acc += num_directors
    actor2nid = {actor: i + acc for i, actor in enumerate(actors)}
    for i, actor in enumerate(actors):
        nid2e[i + acc] = ('actor', actor)
    acc += num_actors
    writer2nid = {writer: i + acc for i, writer in enumerate(writers)}
    for i, writer in enumerate(writers):
        nid2e[i + acc] = ('writer', writer)
    e2nid = {'uid': uid2nid, 'iid': iid2nid, 'gender': gender2nid, 'occ': occ2nid, 'age': age2nid, 'genre': genre2nid,
             'year': year2nid, 'director': director2nid, 'actor': actor2nid, 'writer': writer2nid}

    #########################  create graphs  #########################
    row_idx, col_idx = [], []
    edge_attrs = []

    rating_begin = 0

    print('Creating user property edges...')
    for _, row in tqdm.tqdm(users.iterrows(), total=users.shape[0]):
        uid = row['uid']
        gender = row['gender']
        occ = row['occupation']
        age = row['age']

        u_nid = e2nid['uid'][uid]
        gender_nid = e2nid['gender'][gender]
        row_idx.append(gender_nid)
        col_idx.append(u_nid)
        edge_attrs.append([relations.index('gender'), -1])

        occ_nid = e2nid['occ'][occ]
        row_idx.append(occ_nid)
        col_idx.append(u_nid)
        edge_attrs.append([relation2index['occupation'], -1])

        age_nid = age2nid[age]
        row_idx.append(age_nid)
        col_idx.append(u_nid)
        edge_attrs.append([relation2index['age'], -1])
    rating_begin += 3 * users.shape[0]

    print('Creating item property edges...')
    for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
        iid = row['iid']
        year = row['discretized_year']
        director = row['director']
        actors = row['actor'].split(', ')
        writer = row['writer']

        i_nid = e2nid['iid'][iid]
        y_nid = e2nid['year'][year]
        row_idx.append(y_nid)
        col_idx.append(i_nid)
        edge_attrs.append([relation2index['year'], -1])
        rating_begin += 1

        if director != '':
            d_nid = e2nid['director'][director]
            row_idx.append(d_nid)
            col_idx.append(i_nid)
            edge_attrs.append([relation2index['director'], -1])
            rating_begin += 1

        for actor in actors:
            if actor != '':
                a_nid = e2nid['actor'][actor]
                row_idx.append(a_nid)
                col_idx.append(i_nid)
                edge_attrs.append([relation2index['actor'], -1])
                rating_begin += 1

        if writer != '':
            w_nid = e2nid['writer'][writer]
            row_idx.append(w_nid)
            col_idx.append(i_nid)
            edge_attrs.append([relation2index['writer'], -1])
            rating_begin += 1

        for genre in genres:
            if not row[genre]:
                continue
            g_nid = e2nid['genre'][genre]
            row_idx.append(g_nid)
            col_idx.append(i_nid)
            edge_attrs.append([relation2index['genre'], -1])
            rating_begin += 1

    print('Creating rating property edges...')
    row_idx += [e2nid['uid'][uid] for uid in ratings['uid']]
    col_idx += [e2nid['iid'][iid] for iid in ratings['iid']]
    edge_attrs += [[relation2index['interact'], i] for i in list(ratings['rating'])]

    if not directed:
        print('Creating inverse user property edges...')
        for _, row in tqdm.tqdm(users.iterrows(), total=users.shape[0]):
            uid = row['uid']
            gender = row['gender']
            occ = row['occupation']
            age = row['age']

            u_nid = e2nid['uid'][uid]
            gender_nid = e2nid['gender'][gender]
            row_idx.append(u_nid)
            col_idx.append(gender_nid)
            edge_attrs.append([relations.index('-gender'), -1])

            occ_nid = e2nid['occ'][occ]
            row_idx.append(u_nid)
            col_idx.append(occ_nid)
            edge_attrs.append([relation2index['-occupation'], -1])

            age_nid = age2nid[age]
            row_idx.append(u_nid)
            col_idx.append(age_nid)
            edge_attrs.append([relation2index['-age'], -1])

        print('Creating inverse item property edges...')
        for _, row in tqdm.tqdm(items.iterrows(), total=items.shape[0]):
            iid = row['iid']
            year = row['discretized_year']
            director = row['director']
            actors = row['actor'].split(', ')
            writer = row['writer']

            i_nid = e2nid['iid'][iid]
            y_nid = e2nid['year'][year]
            row_idx.append(i_nid)
            col_idx.append(y_nid)
            edge_attrs.append([relation2index['-year'], -1])

            if director != '':
                d_nid = e2nid['director'][director]
                row_idx.append(i_nid)
                col_idx.append(d_nid)
                edge_attrs.append([relation2index['-director'], -1])

            for actor in actors:
                if actor != '':
                    a_nid = e2nid['actor'][actor]
                    row_idx.append(i_nid)
                    col_idx.append(a_nid)
                    edge_attrs.append([relation2index['-actor'], -1])

            if writer != '':
                w_nid = e2nid['writer'][writer]
                row_idx.append(i_nid)
                col_idx.append(w_nid)
                edge_attrs.append([relation2index['-writer'], -1])

            for genre in genres:
                if not row[genre]:
                    continue
                g_nid = e2nid['genre'][genre]
                row_idx.append(i_nid)
                col_idx.append(g_nid)
                edge_attrs.append([relation2index['-genre'], -1])

    print('Creating inverse rating property edges...')
    row_idx += [e2nid['iid'][iid] for iid in ratings['iid']]
    col_idx += [e2nid['uid'][uid] for uid in ratings['uid']]
    edge_attrs += [[relation2index['-interact'], i] for i in list(ratings['rating'])]

    row_idx = [int(idx) for idx in row_idx]
    col_idx = [int(idx) for idx in col_idx]
    row_idx = np.array(row_idx).reshape(1, -1)
    col_idx = np.array(col_idx).reshape(1, -1)
    edge_index = np.concatenate((row_idx, col_idx), axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    edge_attrs = np.array(edge_attrs)
    edge_attrs = torch.from_numpy(edge_attrs).float()

    #########################  Build masks  #########################
    print('Building masks...')
    rating_mask = torch.ones(ratings.shape[0], dtype=torch.bool)
    if directed:
        rating_edge_mask = torch.cat(
            (
                torch.zeros(rating_begin, dtype=torch.bool),
                rating_mask,
                rating_mask
            ),
            dim=0
        )
    else:
        rating_edge_mask = torch.cat(
            (
                torch.zeros(rating_begin, dtype=torch.bool),
                rating_mask,
                torch.zeros(rating_begin, dtype=torch.bool),
                rating_mask
            ),
            dim=0
        )

    kwargs = {
        'num_nodes': num_nodes, 'edge_index': edge_index, 'edge_attr': edge_attrs,
        'rating_edge_mask': rating_edge_mask,
        'users': users, 'ratings': ratings, 'items': items,
        'relation2index': relation2index, 'index2relation': index2relation, 'num_relations': len(relations),
        'e2nid': e2nid, 'nid2e': nid2e
    }

    if train_ratio is not None:
        train_rating_mask = torch.zeros(ratings.shape[0], dtype=torch.bool)
        test_rating_mask = torch.zeros(ratings.shape[0], dtype=torch.bool)
        train_rating_idx = np.random.choice(range(ratings.shape[0]), int(ratings.shape[0] * train_ratio))
        test_rating_idx = list(set(range(ratings.shape[0])) - set(train_rating_idx))
        train_rating_mask[train_rating_idx] = 1
        test_rating_mask[test_rating_idx] = 1

        if directed:
            train_edge_mask = torch.cat(
                (
                    torch.ones(rating_begin, dtype=torch.bool),
                    train_rating_mask,
                    train_rating_mask)
            )

            test_edge_mask = torch.cat(
                (
                    torch.zeros(rating_begin, dtype=torch.bool),
                    test_rating_mask,
                    test_rating_mask)
            )
        else:
            train_edge_mask = torch.cat(
                (
                    torch.ones(rating_begin, dtype=torch.bool),
                    train_rating_mask,
                    torch.ones(rating_begin, dtype=torch.bool),
                    train_rating_mask)
            )

            test_edge_mask = torch.cat(
                (
                    torch.zeros(rating_begin, dtype=torch.bool),
                    test_rating_mask,
                    torch.zeros(rating_begin, dtype=torch.bool),
                    test_rating_mask)
            )
        kwargs['train_edge_mask'] = train_edge_mask
        kwargs['test_edge_mask'] = test_edge_mask
        if step_length:
            print('Creating path features...')
            path_np = create_path(edge_index[:, train_edge_mask], step_length)
            kwargs['path_np'] = filter_path(path_np)
            kwargs['num_path'] = kwargs['path_np'].shape[1]

        train_user_pos_neg_pair, test_user_pos_neg_pair =  create_user_pos_neg_pair(ratings, train_rating_idx, test_rating_mask, e2nid)
        kwargs['train_user_pos_neg_pair'], kwargs['test_user_pos_neg_pair'] = train_user_pos_neg_pair, test_user_pos_neg_pair
    else:
        if step_length:
            print('Creating path features...')
            path = create_path(edge_index, step_length)
            kwargs['path_np'] = filter_path(path)
            kwargs['num_path'] = kwargs['path_np'].shape[1]

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
        self.num_feat_core = kwargs.get('num_core', 5)
        self.step_length = kwargs.get('step_length', 2)
        self.implicit = kwargs.get('implicit', True)
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
            users, items, ratings = read_ml(self.processed_dir, processed=True)
            users = users.fillna('')
            items = items.fillna('')
            ratings = ratings.fillna('')
        else:
            print('Read from raw data!')
            users, items, ratings = read_ml(unzip_raw_dir, processed=False)

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
            writers_str = items['writer']
            writers = list(itertools.chain.from_iterable([writer_str.split(', ') for writer_str in writers_str]))
            writers_dict = Counter(writers)
            unique_writers = [k for k, v in writers_dict.items() if v > self.num_feat_core]
            writers_str = [', '.join([writer for writer in writer_str.split(', ') if writer in unique_writers]) for writer_str in writers_str]
            directors_str = items['director']
            directors = list(itertools.chain.from_iterable([director_str.split(', ') for director_str in directors_str]))
            directors_dict = Counter(directors)
            writers = list(itertools.chain.from_iterable([writer_str.split(', ') for writer_str in writers_str]))
            unique_directors = {k: v for k, v in directors_dict.items() if v > self.num_feat_core}.keys()
            directors = [director if director in unique_directors else "" for director in directors ]
            actor_strs = [actor_str for actor_str in items.actor.values if actor_str != '']
            actors = [actor_str.split(', ') for actor_str in actor_strs]
            actors = list(itertools.chain.from_iterable(actors))
            actors_dict = Counter(actors)
            unique_actors = {k: v for k, v in actors_dict.items() if v > self.num_feat_core}.keys()
            actor_strs = [[single_actor_str for single_actor_str in actor_str.split(', ') if single_actor_str in unique_actors] for actor_str in actor_strs]
            actor_strs = [', '.join(actor_str) for actor_str in actor_strs]
            items['writer'] = writers_str
            items['director'] = ""
            items['actor'] = ""

            users, items, ratings = reindex_df(users, items, ratings)
            save_df(users, join(self.processed_dir, 'users.pkl'))
            save_df(items, join(self.processed_dir, 'movies.pkl'))
            save_df(ratings, join(self.processed_dir, 'ratings.pkl'))

        if self.debug:
            ratings = ratings.iloc[np.random.choice(range(ratings.shape[0]), int(self.debug * ratings.shape[0]))]
            users = users[users.uid.isin(ratings.uid)]
            items = items[items.iid.isin(ratings.iid)]

        data = convert_2_data(users, items, ratings, self.train_ratio, self.step_length)

        torch.save(self.collate([data]), self.processed_paths[0], pickle_protocol=4)

    def __repr__(self):
        return '{}-{}'.format(self.__class__.__name__, self.name.capitalize())

    def build_suffix(self):
        suffixes = []
        if self.train_ratio is not None:
            suffixes.append('train_{}'.format(self.train_ratio))
        if self.debug:
            suffixes.append('debug_{}'.format(self.debug))
        if self.step_length:
            suffixes.append('path_{}'.format(self.step_length))
        if self.seed is not None:
            suffixes.append('seed_{}'.format(self.seed))
        if not suffixes:
            suffix = ''
        else:
            suffix = '_'.join(suffixes)
        return '_' + suffix


if __name__ == '__main__':
    import os.path as osp
    root = osp.join('.', 'tmp', 'ml')
    name = '1m'
    debug = 0.01
    dataset = MovieLens(root=root, name='1m', debug=debug, train_ratio=0.8)

