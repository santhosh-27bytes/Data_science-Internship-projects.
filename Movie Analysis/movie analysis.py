import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

movies = pd.read_csv(r"C:\Users\santh\OneDrive\Desktop\intern\task 1\movies.csv")

movies['genres'] = movies['genres'].str.replace('|', ' ')

cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['genres'])

similarity = cosine_similarity(count_matrix)

index = movies[movies['title'] == 'Toy Story (1995)'].index[0]

scores = list(enumerate(similarity[index]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)

for i in scores[1:6]:
    print(movies.iloc[i[0]].title)
