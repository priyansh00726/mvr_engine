#%%

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs
import pandas as pd

cv = CountVectorizer()

df = pd.read_csv("./movie_metadata.csv")
# df.head

features = ['actor_1_name', 'actor_2_name', 'genres', 'director_name', 'plot_keywords', 'movie_title']
for feature in features:
    df[feature] = df[feature].fillna("")
    df[feature] = df[feature].str.strip()


def combine_feature(row, feat):
    row_res = ""
    for feature in feat:
        row_res = row_res + row[feature] + " " 
    return row_res


df["combined_feature"] = df.apply(lambda row: combine_feature(row, feat=features), axis=1)
# df.iloc[0]["combined_feature"]
# df.head()

count_matrix = cv.fit_transform(df["combined_feature"])
# count_matrix.toarray()
cosine_model = pd.DataFrame(cs(count_matrix), index=df.movie_title, columns=df.movie_title)
cosine_model.index


def make_recommendations(movie_user_likes):
    return cosine_model[movie_user_likes].sort_values(ascending=False)[:20]

make_recommendations("The Dark Knight")
