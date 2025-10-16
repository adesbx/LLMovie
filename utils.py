import json
import re
import difflib
import zipfile
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_path_movie = "./dataset/movies.csv"
df_path_rates = "./dataset/ratings.csv"
letterbox_zip_path = "./dataset/letterboxd-chewbou-2025-10-16-13-11-utc.zip"


def extract_ratings(csv_file: str) -> list[dict]: 
    df = pd.read_csv(csv_file)
    
    movies = []

    for _, row in df.iterrows():
        movies.append(
            {
                "title": row["Name"],
                "year": row["Year"],
                "rate": row["Rating"],
                "uri_lb": row["Letterboxd URI"]
            }
        )

    return movies

def extract_watchlist_or_likes(csv_file: str) -> list[dict]: 
    df = pd.read_csv(csv_file)
    
    movies = []

    for _, row in df.iterrows():
        movies.append(
            {
                "title": row["Name"],
                "year": row["Year"],
                "uri_lb": row["Letterboxd URI"]
            }
        )

    return movies

def find_by_id(id_movie: int):
    df = pd.read_csv(df_path_movie)
    t = df.loc[df['movieId'] == id_movie]
    return t["title"], t["genres"]

def find_by_title(title: str, year: int, df: pd.DataFrame):
   subset = df[df["year"] == year] 
   best_id = 0
   best_ratio = 0.0
   for _, row in subset.iterrows():
      ratio = difflib.SequenceMatcher(None, row["clean_title"], title).ratio()
      if ratio > 0.9:
         best_id = row["movieId"]
         best_ratio = ratio
         TEMP = row["title"]

   # if best_id == 0:
   #    print(title + " not found")
   # else:
   #    print(title + " found in " + TEMP)      
   return best_id

def find_only_by_title(title: str): # ne retourne pas forcément le bon
    df = pd.read_csv(df_path_movie)
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    best_id = 0
    best_ratio = 0.0
    for _, row in df.iterrows():
        ratio = difflib.SequenceMatcher(None, row["clean_title"], title).ratio()
        if ratio > 0.9:
            best_id = row["movieId"]
        
    return best_id

def find_all(rates : list):
    sum_not_found = 0
    letterboxd_user = {}
    df = pd.read_csv(df_path_movie)
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype(float)
    for dico in rates:
        id = find_by_title(dico["title"], dico["year"], df)
        if id != 0:
            letterboxd_user.update({id: dico["rate"]})
        else:
            sum_not_found = sum_not_found + 1

    return letterboxd_user, sum_not_found

def upload_data(file):
    with zipfile.ZipFile(file, "r") as zip:
       with zip.open("ratings.csv") as csv_file:
           rates = extract_ratings(csv_file)

       with zip.open("likes/films.csv") as csv_file:
           likes = extract_watchlist_or_likes(csv_file)

    return rates, likes

def caculate_factors(): # TODO: faire une classe qui calcule tout
    df = pd.read_csv(df_path_rates)
    #init
    users, users_id = np.unique(df["userId"], return_inverse=True)
    movies, movies_id = np.unique(df["movieId"], return_inverse=True)
    df['u'] = users_id
    df['m'] = movies_id

    n_users = len(users)
    n_movies = len(movies)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    r_train = train_df["rating"].values
    u_train = train_df["u"].values
    m_train = train_df["m"].values

    mu = r_train.mean()
    lambda_reg = 10.0

    #calcul pour film
    item_sum = np.zeros(n_movies)
    item_count = np.zeros(n_movies)

    for mi, ri in zip(m_train, r_train):
        item_sum[mi] += ri
        item_count[mi] += 1
    item_bias = np.zeros(len(item_count))
    mask = item_count > 0
    item_bias[mask] = (item_sum[mask] - item_count[mask] * mu) / (item_count[mask] + lambda_reg)

    #calcul pour user
    user_sum = np.zeros(n_users)
    user_count = np.zeros(n_users)

    for ui, mi, ri in zip(u_train, m_train, r_train):
        user_sum[ui] += (ri - mu - item_bias[mi])
        user_count[ui] += 1
    user_bias = np.zeros(len(user_count))
    mask_u = user_count > 0
    user_bias[mask_u] = user_sum[mask_u] / (user_count[mask_u] + lambda_reg)

    residuals = r_train - (mu - user_bias[u_train] + item_bias[m_train])
    R_resid = csr_matrix((residuals, (u_train, m_train)), shape=(n_users, n_movies))

    k = 50 
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
    user_factors = svd.fit_transform(R_resid)  
    item_factors = svd.components_.T  

    return svd, user_factors, item_factors, users, movies, mu, user_bias, item_bias

def predict_scores_for_user(user_idx, mu, user_bias, item_bias, user_factors, item_factors):
    baseline_u = mu + user_bias[user_idx] + item_bias  
    interaction = user_factors[user_idx].dot(item_factors.T)  
    return baseline_u + interaction

def top_k_recs(user_idx, mu, user_bias, item_bias, user_factors, item_factors, df, K=10):
    scores = predict_scores_for_user(user_idx, mu, user_bias, item_bias, user_factors, item_factors)
    seen = set(df.loc[df['u'] == user_idx, 'm'].values)

    if len(seen) > 0:
        scores[list(seen)] = -np.inf
    topk_idx = np.argpartition(-scores, K)[:K]
    topk_sorted = topk_idx[np.argsort(-scores[topk_idx])]
    return topk_sorted, scores[topk_sorted]

def recommend(user_id, mu, user_bias, item_bias, user_factors, item_factors, users, movies, df, K=10):
    if user_id not in users:
        print(f"Utilisateur {user_id} inconnu (cold-start).")
        return []
    
    user_idx = np.where(users == user_id)[0][0]

    scores = predict_scores_for_user(user_idx, mu, user_bias, item_bias, user_factors, item_factors)

    seen = set(df.loc[df['u'] == user_idx, 'm'].values)
    if len(seen) > 0:
        scores[list(seen)] = -np.inf

    topk_idx = np.argpartition(-scores, K)[:K]
    topk_sorted = topk_idx[np.argsort(-scores[topk_idx])]

    top_movie_ids = movies[topk_sorted]

    results = [(int(mid), float(scores[idx])) for mid, idx in zip(top_movie_ids, topk_sorted)]
    return results

def recommend_new_user(ratings_dict, svd, mu, movies, item_bias, item_factors, n_movies, K=10):
    m_indices = []
    resids = []
    for mid, r in ratings_dict.items():
        if mid in movies:  
            m_idx = np.where(movies == mid)[0][0]
            resid = r - (mu + item_bias[m_idx])
            m_indices.append(m_idx)
            resids.append(resid)

    if not m_indices:
        print("Aucun film du nouvel utilisateur n'existe dans le dataset.")
        return []

    row = np.zeros((1, n_movies))
    row[0, m_indices] = resids

    user_vec = svd.transform(row)  # shape (1, k)

    baseline = mu + item_bias
    interaction = user_vec.dot(item_factors.T).flatten()
    scores = baseline + interaction

    scores[m_indices] = -np.inf

    topk_idx = np.argpartition(-scores, K)[:K]
    topk_sorted = topk_idx[np.argsort(-scores[topk_idx])]
    top_movie_ids = movies[topk_sorted]

    return [(int(mid), float(scores[idx])) for mid, idx in zip(top_movie_ids, topk_sorted)]

def recommend_similar_movie(movie_id, movies, min_rates=4.0, K=3):
    df = pd.read_csv(df_path_rates)

    users_likes = df.loc[
        (df['movieId'] == movie_id) & (df['rating'] >= min_rates),
        'userId'
    ].unique()

    if len(users_likes) == 0:
        return "Aucun utilisateurs n'a aimé ce film"

    similar = df[df['userId'].isin(users_likes)]
    similar = similar[similar['movieId'] != movie_id]

    if similar.empty:
        return "Aucun utilisateurs n'a aimé d'autres films"
    
    movie_score = similar.groupby('movieId').agg(
        count=('rating', 'count'),
        mean_rating=('rating', 'mean')
    )

    movie_score['score'] = movie_score['mean_rating'] * (
        movie_score['count'] / movie_score['count'].max()
    )

    top_k = movie_score.sort_values('score', ascending=False).head(K).reset_index()
    top_k = top_k[top_k['movieId'].isin(movies)]

    return top_k

def choose_method(fun: str, question: str):
    to_return = ""
    print(fun)
    fun = fun.strip()
    split = fun.split(",")
    fun = split[0]
    movie_name = split[1].strip()
    match fun:
        case "recommend_by_user": 
            # print("recommand by user")
            svd, user_factors, item_factors, users, movies, mu, user_bias, item_bias = caculate_factors()
            rates, likes = upload_data(letterbox_path)

            user, _ = find_all(rates) # revoir les nons pris en compte
            l = recommend_new_user(user, svd, mu, movies, item_bias, item_factors, len(movies), 3)

            l_recommend = []
            for id, _ in l:
                title, _ = find_by_id(id)
                l_recommend.append(title.iloc[0])
                
            to_return = l_recommend
        case "recommend_by_movie": 
            _, _, _, _, movies, _, _, _ = caculate_factors()
            # print(movie_name)
            id_movie = find_only_by_title(movie_name)
            print(id_movie)
            if id_movie != 0:
                l = recommend_similar_movie(id_movie, movies)
                l = l['movieId']
                l_recommend = []
                for id in l:
                    title, _ = find_by_id(id)
                    l_recommend.append(title.iloc[0])
                    
                to_return = l_recommend
            else:
                to_return = question
            print(to_return)
        case "simple_recommend": 
            to_return = question
        case _:
            print("Aucun cas trouvé")

    # print("before return")
    # print(to_return)
    return to_return
