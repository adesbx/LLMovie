import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

df_path_movie = "./dataset/movies.csv"
df_path_rates = "./dataset/ratings.csv"

class Factor:
    def __init__(self):
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

        self.svd = svd
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.users = users
        self.movies = movies
        self.mu = mu
        self.user_bias = user_bias
        self.item_bias = item_bias