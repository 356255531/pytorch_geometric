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


def read_ml(dir, processed=False):
    """
    Read the movielens dataset from .dat file
    :param dir: the path to raw files (users.dat, movies.dat, ratings.dat)
    :param debug: the portion of ratings userd, float
    :return: users, movies, ratings, pandas.DataFrame
    """

    if processed:
        users = pd.read_csv(join(dir, 'users.pkl'))
        movies = pd.read_csv(join(dir, 'movies.pkl'))
        ratings = pd.read_csv(join(dir, 'ratings.pkl'))
    else:
        users = []
        with open(join(dir, 'users.dat')) as f:
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

        movies = []
        # read movies
        with open(join(dir, 'movies.dat'), encoding='latin1') as f:
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


            try:
                movie_url = "http://www.omdbapi.com/?" + "t=" + title + "&y=" + str(year) + "&apikey=" + apikey
                r = requests.get(movie_url)
                movie_info_dic = json.loads(r.text)
            except:
                try:
                    movie_url = "http://www.omdbapi.com/?" + "t=" + title + "&apikey=" + apikey
                    r = requests.get(movie_url)
                    movie_info_dic = json.loads(r.text)
                except:
                    movie_info_dic = dict()
            director = movie_info_dic.get('Director', '')
            actor = movie_info_dic.get('Actors', '')
            if actor == '':
                print('*************')
            else:
                print(actor)
            writer = movie_info_dic.get('Writer', '')

            director_list.append(director)
            actor_list.append(actor)
            writer_list.append(writer)

        movies['director'] = director_list
        movies['actor'] = actor_list
        movies['writer'] = writer_list

        ratings = []
        with open(join(dir, 'ratings.dat')) as f:
            for line in f:
                user_id, movie_id, rating, timestamp = [int(_) for _ in line.split('::')]
                ratings.append({
                    'uid': user_id,
                    'iid': movie_id,
                    'rating': rating - 1,
                    'timestamp': timestamp,
                })
        ratings = pd.DataFrame(ratings)

    return users, movies, ratings
