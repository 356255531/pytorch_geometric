import re
import numpy as np
import pandas as pd
from os.path import join
import requests
import json
import tqdm

apikey = 'e760129c'


def get_director_actors(movie_title, movie_year):
    movie_url = "http://www.omdbapi.com/?" + "t=" + movie_title + "&y=" + movie_year + "&apikey=" + apikey
    try:
        r = requests.get(movie_url)
        movie_info_dic = json.loads(r.text)
    except:
        return '', ''

    director = movie_info_dic.get('Director', '')
    actors = movie_info_dic.get('Actors', '')

    return director, actors


def get_poster(movie_title, movie_year):
    movie_url = "http://www.omdbapi.com/?" + "t=" + movie_title + "&y=" + movie_year + "&apikey=" + apikey
    r = requests.get(movie_url)
    movie_info_dic = json.loads(r.text)
    poster = movie_info_dic['Poster']
    return poster


def read_ml(raw_dir, debug=None):
    """
    Read the movielens dataset from .dat file
    :param raw_dir: the path to raw files (users.dat, movies.dat, ratings.dat)
    :param debug: the portion of ratings userd, float
    :return: users, movies, ratings, pandas.DataFrame
    """
    users = []
    with open(join(raw_dir, 'users.dat')) as f:
        for l in f:
            id_, gender, age, occupation, zip_ = l.strip().split('::')
            users.append({
                'uid': int(id_),
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'zip': zip_,
            })
    users = pd.DataFrame(users)

    movies = []
    # read movies
    with open(join(raw_dir, 'movies.dat'), encoding='latin1') as f:
        for l in f:
            id_, title, genres = l.strip().split('::')
            genres_set = set(genres.split('|'))

            # extract year
            assert re.match(r'.*\([0-9]{4}\)$', title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {'iid': int(id_), 'title': title, 'year': year}
            for g in genres_set:
                data[g] = True
            movies.append(data)
    movies = (
        pd.DataFrame(movies)
            .fillna(False)
            .astype({'year': 'category'}))
    movie_titles, movie_years = movies.title.values, movies.year.values
    # pbar = tqdm.tqdm(zip(movie_titles, movie_years), total=movies.shape[0])
    # print('Getting directors and actors...')
    # director_actors = [get_director_actors(title, str(year)) for title, year in pbar]
    # directors = [i[0] for i in director_actors]
    # actors = [i[1] for i in director_actors]
    # movies['director'] = directors
    # movies['actors'] = actors

    ratings = []
    with open(join(raw_dir, 'ratings.dat')) as f:
        for l in f:
            user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
            ratings.append({
                'uid': user_id,
                'iid': movie_id,
                'rating': rating - 1,
                'timestamp': timestamp,
            })
    ratings = pd.DataFrame(ratings)

    if debug:
        df_idx = np.random.choice(np.arange(ratings.shape[0]), int(ratings.shape[0] * debug))
        ratings = ratings.iloc[df_idx]

    return users, movies, ratings
