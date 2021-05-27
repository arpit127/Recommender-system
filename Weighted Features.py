import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


###### helper functions. Use them when needed #######
def get_title_from_index(index):
    return df["original_title"].iloc[int(index)]


def get_row_from_title(title):
    return df[df["title"] == title].iloc[0]


##################################################

##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")
movies = df['title']
movies = movies.fillna('', inplace=False)
for i in range(movies.shape[0]):
    movies.iloc[i] = movies.iloc[i].lower()

df['title'] = movies
# print df.columns
##Step 2: Select Features

features = ['keywords', 'cast', 'genres', 'director']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')


def calculate_similarity_score(movie, candidate, weights):
    scores = []
    for feature in features:
        f1 = movie[feature]
        f2 = candidate[feature]

        # Step 4: Create count matrix from this new combined column
        cv = CountVectorizer()
        count_matrix = cv.fit_transform([f1, f2])

        # Step 5: Compute the Cosine Similarity based on the count_matrix
        cosine_sim = cosine_similarity(count_matrix[0], count_matrix[1])[0][0]
        scores.append(cosine_sim)
    scores = np.array(scores)
    final_score = np.dot(scores, weights.T)
    return final_score


inp = input("Enter your favourite movie name: ")
movie_user_likes = inp.lower()

# Step 6: Get index of this movie from its title
movie_input = get_row_from_title(movie_user_likes)


default_rank = 100
ranks=[]
weights = [[0, 0.33, 0.33, 0.33], [0.33, 0, 0.33, 0.33], [0.33, 0.33, 0, 0.33], [0.33, 0.33, 0.33, 0],
           [0.25, 0.25, 0.25, 0.25], [0.4, 0.1, 0.4, 0.1], [0.1, 0.4, 0.1, 0.4]]
for weight in weights:
    similarity_scores = []
    for i in range(df.shape[0]):
        candidate = df.iloc[i]
        similarity_score = calculate_similarity_score(movie_input, candidate, np.array(weight))
        similarity_scores.append([i, similarity_score])

        # Step 7: Get a list of similar movies in descending order of similarity score
    sorted_similar_movies = sorted(similarity_scores, key=lambda x:x[1], reverse=True)

    # Step 8: Print titles of first 50 movies
    j = 0
    print("Top 5 movies that you might love to watch: ")
    for element in sorted_similar_movies:
        print(get_title_from_index(element[0]))
        j += 1
        if j > 5:
            break
    print("============================================")

    count = 0
    average_rank = 0
    for m in range(10):
        movie = get_title_from_index(sorted_similar_movies[m][0])
        if movie in ["Superman", "Batman v Superman: Dawn of Justice",
                     "Watchmen"]:
            count += 1
            average_rank += m+1
    if count == 2:
        average_rank += default_rank
    elif count == 1:
        average_rank += 2*default_rank
    elif count == 0:
        average_rank += 3*default_rank
    average_rank /= 3.0
    ranks.append([weight, average_rank])
    # break
'''   
# Step 8: Print titles of first 50 movies
i = 0
print("Top 5 movies that you might love to watch: ")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i = i + 1
    if i > 10:
        break
'''
