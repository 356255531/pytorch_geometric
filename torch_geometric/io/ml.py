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
        for line in f:
            id_, gender, age, occupation, zip_ = line.strip().split('::')
            users.append({
                'uid': int(id_),
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'zip': zip_,
            })
    users = pd.DataFrame(users)

    try:
        movies = pd.read_csv(join(raw_dir, 'new_movies.dat'))
    except:
        movies = []

        # read movies
        with open(join(raw_dir, 'movies.dat'), encoding='latin1') as f:
            for line in f:
                id_, title, genres = line.strip().split('::')
                genres_set = set(genres.split('|'))

                # extract year
                assert re.match(r'.*\([0-9]{4}\)$', title)
                year = title[-5:-1]
                title = title[:-6].strip()

                data = {'iid': int(id_), 'title': title, 'year': int(year)}
                for g in genres_set:
                    data[g] = True
                movies.append(data)
        movies = (
            pd.DataFrame(movies)
                .fillna(False)
                .astype({'year': 'category'}))

        apikey = ''
        key1 = 'e760129c'
        key2 = 'e44e5305'
        key3 = '8403a97b'
        key4 = '192c6b0e'

        director_list = []
        actor_list = []
        writer_list = []

        pbar = tqdm.tqdm(zip(movies.title, movies.year), total=movies.shape[0])
        for i, (title, year) in enumerate(pbar):
            if i in range(0, 1000):
                apikey = key1
            if i in range(1000, 2000):
                apikey = key2
            if i in range(2000, 3000):
                apikey = key3
            if i in range(3000, 4000):
                apikey = key4

            movie_url = "http://www.omdbapi.com/?" + "t=" + title + "&y=" + str(year) + "&apikey=" + apikey
            # print('i=' + str(i) + ',apikey=' + apikey )
            try:
                r = requests.get(movie_url)
                movie_info_dic = json.loads(r.text)
                director = movie_info_dic.get('Director', '')
                actor = movie_info_dic.get('Actors', '')
                writer = movie_info_dic.get('Writer', '')
            except:
                director = ''
                actor = ''
                writer = ''
            director_list.append(director)
            actor_list.append(actor)
            writer_list.append(writer)

        movies['director'] = director_list
        movies['actor'] = actor_list
        movies['writer'] = writer_list

    ratings = []
    with open(join(raw_dir, 'ratings.dat')) as f:
        for line in f:
            user_id, movie_id, rating, timestamp = [int(_) for _ in line.split('::')]
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
