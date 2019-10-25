import re
import pandas as pd
from os.path import join


def read_ml(raw_dir, debug=False):

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
        ratings = ratings[:10000]

    return users, movies, ratings
