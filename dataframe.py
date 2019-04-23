import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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


print(metadata['overview'].head())


tfidf = TfidfVectorizer(stop_words='english')


metadata['overview'] = metadata['overview'].fillna(' ')


tfidf_matrix = tfidf.fit_transform(metadata['overview'])

'''print(tfidf_matrix.shape)'''


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()



def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return print(metadata['title'].iloc[movie_indices])


get_recommendations('The Dark Knight Rises')