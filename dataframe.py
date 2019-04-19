import pandas as pd

metadata = pd.read_csv('movies_metadata.csv', low_memory = False)

metadata.head(3)

c = metadata['vote_average'].mean()

print(c)

m = metadata['vote_count'].quantile(0.90)

print(m)

q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print(q_movies.shape)


def weighted_rating(x, m=m, c=c):
    v = x['vote_count']
    r = x['vote_average']

    return (v/(v+m) * r) + (m/(m+v) * c)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)

print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))




