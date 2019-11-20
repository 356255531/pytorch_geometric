import torch
import numpy as np
import random as rd
import pickle
import pandas as pd


from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_lastfm
from torch_geometric.data import Data, extract_zip
from torch_geometric.utils import get_sec_order_edge


def reindex_df(
        raw_uids, raw_aids, raw_tids,
        artists, tags, user_artists, bi_user_friends, user_taggedartists):
    """
    :param uids:
    :param aids:
    :param tids:
    :param tagged_artists:
    :param tags:
    :param user_artists:
    :param bi_user_friends:
    :param user_taggedartists:
    :return:
    """
    uids = np.arange(raw_uids.shape[0])
    aids = np.arange(raw_aids.shape[0])
    tids = np.arange(raw_tids.shape[0])

    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(raw_uids, uids)}
    raw_aid2aid = {raw_aid: aid for raw_aid, aid in zip(raw_aids, aids)}
    raw_tid2tid = {raw_tid: tid for raw_tid, tid in zip(raw_tids, tids)}

    print('reindex artist index of artists...')
    artists_aids = np.array(artists.aid, dtype=np.int)
    artists_aids = [raw_aid2aid[aid] for aid in artists_aids]
    artists.loc[:, 'aid'] = artists_aids

    print('reindex tag index of tags...')
    tags_tids = np.array(tags.tid, dtype=np.int)
    tags_tids = [raw_tid2tid[tid] for tid in tags_tids]
    tags.loc[:, 'tid'] = tags_tids

    print('reindex user and artist index of user_artists...')
    user_artists_uids = np.array(user_artists.uid, dtype=np.int)
    user_artists_aids = np.array(user_artists.aid, dtype=np.int)
    user_artists_uids = [raw_uid2uid[uid] for uid in user_artists_uids]
    user_artists_aids = [raw_aid2aid[aid] for aid in user_artists_aids]
    user_artists.loc[:, 'uid'] = user_artists_uids
    user_artists.loc[:, 'aid'] = user_artists_aids

    print('reindex user and friends index of bi_user_friends...')
    bi_user_friends_uids = np.array(bi_user_friends.uid, dtype=np.int)
    bi_user_friends_fids = np.array(bi_user_friends.fid, dtype=np.int)
    bi_user_friends_uids = [raw_uid2uid[uid] for uid in bi_user_friends_uids]
    bi_user_friends_fids = [raw_uid2uid[fid] for fid in bi_user_friends_fids]
    bi_user_friends.loc[:, 'uid'] = bi_user_friends_uids
    bi_user_friends.loc[:, 'fid'] = bi_user_friends_fids

    print('reindex user, artist and tag index of bi_user_friends...')
    user_taggedartists_uids = np.array(user_taggedartists.uid, dtype=np.int)
    user_taggedartists_aids = np.array(user_taggedartists.aid, dtype=np.int)
    user_taggedartists_tids = np.array(user_taggedartists.tid, dtype=np.int)
    user_taggedartists_uids = [raw_uid2uid[uid] for uid in user_taggedartists_uids]
    user_taggedartists_aids = [raw_aid2aid[aid] for aid in user_taggedartists_aids]
    user_taggedartists_tids = [raw_tid2tid[tid] for tid in user_taggedartists_tids]
    user_taggedartists.loc[:, 'uid'] = user_taggedartists_uids
    user_taggedartists.loc[:, 'aid'] = user_taggedartists_aids
    user_taggedartists.loc[:, 'tid'] = user_taggedartists_tids
    return uids, aids, tids, artists, tags, user_artists, bi_user_friends, user_taggedartists


def convert_2_data(
        uids, aids, tids,
        artists, tags, user_artists, bi_user_friends, user_taggedartists,
        train_ratio, sec_order):
    """
    n_nodes = n_users + n_artists + n_tags

    """
    n_users = uids.shape[0]
    n_artists = aids.shape[0]
    n_tags = tids.shape[0]

    relations = ['friendship', 'tag', 'interaction', '-friendship', '-tag', '-interaction']

    # Bulid node id
    num_nodes = n_users + n_artists + n_tags
    num_relations = len(relations)

    # Build property2id map
    user_node_id_map = {i: i for i in uids}
    artist_node_id_map = {i: n_users + i for i in aids}
    tag_node_id_map = {i: n_users + n_artists + i for i in tids}

    # Start creating edges
    row_idx, col_idx = [], []
    edge_attrs = []

    print('Creating friendship edges...')
    uids = list(bi_user_friends['uid'].values)
    fids = list(bi_user_friends['fid'].values)
    user_nids = [user_node_id_map[uid] for uid in uids]
    friends_nids = [user_node_id_map[fid] for fid in fids]
    row_idx += user_nids
    col_idx += friends_nids
    edge_attrs += [[-1, relations.index('friendship')] for i in range(bi_user_friends.shape[0])]

    print('Creating artist tags property edges...')
    artists_tags = user_taggedartists[['aid', 'tid']].drop_duplicates()
    aids = list(artists_tags['aid'].values)
    tids = list(artists_tags['tid'].values)
    artist_nids = [artist_node_id_map[aid] for aid in aids]
    tag_nids = [tag_node_id_map[tid] for tid in tids]
    row_idx += artist_nids
    col_idx += tag_nids
    edge_attrs += [[-1, relations.index('tag')] for i in range(artists_tags.shape[0])]

    print('Creating listen property edges...')
    uids = list(user_artists['uid'].values)
    aids = list(user_artists['aid'].values)
    user_nids = [user_node_id_map[uid] for uid in uids]
    artist_nids = [artist_node_id_map[aid] for aid in aids]
    row_idx += user_nids
    col_idx += artist_nids
    edge_attrs += [[1, relations.index('interaction')] for i in range(user_artists.shape[0])]

    print('Building masks...')
    rating_mask = torch.ones(user_artists.shape[0], dtype=torch.bool)
    rating_edge_mask = torch.cat(
        (
            torch.zeros(bi_user_friends.shape[0] + artists_tags.shape[0], dtype=torch.bool),
            rating_mask,
            torch.zeros(artists_tags.shape[0], dtype=torch.bool),
            rating_mask),
    )
    if train_ratio is not None:
        train_rating_mask = torch.zeros(user_artists.shape[0], dtype=torch.bool)
        test_rating_mask = torch.ones(user_artists.shape[0], dtype=torch.bool)
        train_rating_idx = rd.sample([i for i in range(user_artists.shape[0])], int(user_artists.shape[0] * train_ratio))
        train_rating_mask[train_rating_idx] = 1
        test_rating_mask[train_rating_idx] = 0

        train_edge_mask = torch.cat(
            (
                torch.ones(bi_user_friends.shape[0] + artists_tags.shape[0], dtype=torch.bool),
                train_rating_mask,
                torch.ones(artists_tags.shape[0], dtype=torch.bool),
                train_rating_mask)
        )

        test_edge_mask = torch.cat(
            (
                torch.ones(bi_user_friends.shape[0] + artists_tags.shape[0], dtype=torch.bool),
                test_rating_mask,
                torch.ones(artists_tags.shape[0], dtype=torch.bool),
                test_rating_mask)
        )

    print('Creating reverse artist tags property edges...')
    aids = list(artists_tags['aid'].values)
    tids = list(artists_tags['tid'].values)
    artist_nids = [artist_node_id_map[aid] for aid in aids]
    tag_nids = [tag_node_id_map[tid] for tid in tids]
    col_idx += artist_nids
    row_idx += tag_nids
    edge_attrs += [[-1, relations.index('-tag')] for i in range(artists_tags.shape[0])]

    print('Creating reverse listen property edges...')
    uids = list(user_artists['uid'].values)
    aids = list(user_artists['aid'].values)
    user_nids = [user_node_id_map[uid] for uid in uids]
    artist_nids = [artist_node_id_map[aid] for aid in aids]
    col_idx += user_nids
    row_idx += artist_nids
    edge_attrs += [[1, relations.index('-interaction')] for i in range(user_artists.shape[0])]

    row_idx = [int(idx) for idx in row_idx]
    col_idx = [int(idx) for idx in col_idx]
    row_idx = np.array(row_idx).reshape(1, -1)
    col_idx = np.array(col_idx).reshape(1, -1)
    edge_index = np.concatenate((row_idx, col_idx), axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    edge_attrs = np.array(edge_attrs)
    edge_attrs = torch.from_numpy(edge_attrs).long()

    kwargs = {
        'num_nodes': num_nodes, 'num_relations': num_relations,
        'edge_index': edge_index, 'edge_attr': edge_attrs,
        'rating_edge_mask': rating_edge_mask,
        'tags': tags, 'user_artists': user_artists, 'artists': artists,
        'user_node_id_map': user_node_id_map,
        'artist_node_id_map': artist_node_id_map, 'tag_node_id_map': tag_node_id_map
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


class LastFM(InMemoryDataset):
    url = 'http://files.grouplens.org/datasets/hetrec2011/'

    def __init__(self,
                 root,
                 name,
                 sec_order=False,
                 num_cores=10,
                 num_tag_cores=10,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.name = name.lower()
        assert self.name in ['2k']
        self.num_cores = num_cores
        self.num_tag_cores = num_tag_cores
        self.sec_order = sec_order

        self.train_ratio = kwargs.get('train_ratio', None)
        self.debug = kwargs.get('debug', False)
        self.seed = kwargs.get('seed', None)
        self.suffix = self.build_suffix()
        super(LastFM, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        print('Graph params: {}'.format(self.data))

        print('Dataset loaded!')

    @property
    def raw_file_names(self):
        return 'hetrec2011-lastfm-{}.zip'.format(self.name.lower())

    @property
    def processed_file_names(self):
        return ['data{}.pt'.format(self.suffix)]

    def download(self):
        path = download_url(self.url + self.raw_file_names, self.raw_dir)

        extract_zip(path, self.raw_dir)

    def process(self):
        unzip_raw_dir = self.raw_dir

        # read files
        artists, tags, user_artists, user_taggedartists, bi_user_friends = read_lastfm(unzip_raw_dir, self.debug)

        # remove duplications
        artists = artists.drop_duplicates()
        tags = tags.drop_duplicates()
        user_artists = user_artists.drop_duplicates()
        user_taggedartists = user_taggedartists.drop_duplicates()
        bi_user_friends = bi_user_friends.drop_duplicates()

        # Remove the interactions less than num_cores, and rebuild users and artists df
        user_artists = user_artists[user_artists.listen_count > self.num_cores]
        aids = user_artists.aid.drop_duplicates().sort_values()
        artists = artists[artists.aid.isin(aids)]
        aids = artists.aid.drop_duplicates().sort_values()
        user_artists = user_artists[user_artists.aid.isin(aids)]
        uids = user_artists.uid.drop_duplicates().sort_values()

        # Remove the friendship not in uids
        bi_user_friends = bi_user_friends[bi_user_friends.uid.isin(uids) & bi_user_friends.fid.isin(uids)]
        bi_user_friends = bi_user_friends[bi_user_friends.uid != bi_user_friends.fid]

        # Remove the sparse tags from tags and user_taggedartists dataframe
        tag_count = user_taggedartists.tid.value_counts()
        tag_count.name = 'tag_count'
        user_taggedartists = user_taggedartists.join(tag_count, on='tid')
        user_taggedartists = user_taggedartists[user_taggedartists.tag_count > self.num_tag_cores]
        user_taggedartists = user_taggedartists[user_taggedartists.uid.isin(uids) & user_taggedartists.aid.isin(aids)]
        tids = user_taggedartists.tid.drop_duplicates().sort_values()
        tags = tags[tags.tid.isin(tids)]
        tids = tags.tid.drop_duplicates().sort_values()
        user_taggedartists = user_taggedartists[user_taggedartists.tid.isin(tids)]

        uids, aids, tids, artists, tags, user_artists, bi_user_friends, user_taggedartists = \
            reindex_df(uids, aids, tids, artists, tags, user_artists, bi_user_friends, user_taggedartists)

        data = convert_2_data(
            uids, aids, tids, artists, tags, user_artists, bi_user_friends, user_taggedartists,
            self.train_ratio, self.sec_order
        )

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
