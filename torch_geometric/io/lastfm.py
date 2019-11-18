import re
import numpy as np
import pandas as pd
from os.path import join


def read_lastfm(raw_dir, debug=None):
    """
    Read the lastfm dataset from .dat file
    :param raw_dir: the path to raw files (users.dat, movies.dat, ratings.dat)
    :param debug: the portion of ratings userd, float
    :return: artists, tags, user_artists, user_taggedartists, user_friends, pandas.Dataframe
    """
    artists = []
    with open(join(raw_dir, 'artists.dat'), encoding='latin1') as f:
        i = True
        for line in f:
            if i:
                i = False
                continue
            try:
                aid, name, url, pict_url = line.strip().split('\t')
            except:
                continue
            artists.append({
                'aid': int(aid),
                'name': name,
                'url': url,
                'pict_url': pict_url
            })
    artists = pd.DataFrame(artists)

    tags = []
    with open(join(raw_dir, 'tags.dat'), encoding='latin1') as f:
        i = True
        for line in f:
            if i:
                i = False
                continue
            tid, tag = line.strip().split('\t')
            tags.append({
                'tid': int(tid),
                'tag': tag
            })
    tags = pd.DataFrame(tags)

    user_artists = []
    with open(join(raw_dir, 'user_artists.dat'), encoding='latin1') as f:
        i = True
        for line in f:
            if i:
                i = False
                continue
            uid, aid, listen_count = line.strip().split('\t')
            user_artists.append({
                'uid': int(uid),
                'aid': int(aid),
                'listen_count': int(listen_count),
            })
    user_artists = pd.DataFrame(user_artists)

    user_taggedartists = []
    with open(join(raw_dir, 'user_taggedartists.dat'), encoding='latin1') as f:
        i = True
        for line in f:
            if i:
                i = False
                continue
            uid, aid, tid, dd, mm, yy = line.strip().split('\t')
            user_taggedartists.append({
                'uid': int(uid),
                'aid': int(aid),
                'tid': int(tid),
                'dd': int(dd),
                'mm': int(mm),
                'yy': int(yy),
            })
    user_taggedartists = pd.DataFrame(user_taggedartists)

    user_friends = []
    with open(join(raw_dir, 'user_friends.dat'), encoding='latin1') as f:
        i = True
        for line in f:
            if i:
                i = False
                continue
            uid, fid = line.strip().split('\t')
            user_friends.append({'uid': int(uid), 'fid': int(fid)})
    user_friends = pd.DataFrame(user_friends)

    if debug:
        df_idx = np.random.choice(np.arange(user_artists.shape[0]), int(user_artists.shape[0] * debug))
        user_artists = user_artists.iloc[df_idx]

    return artists, tags, user_artists, user_taggedartists, user_friends

