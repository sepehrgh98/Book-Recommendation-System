# from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from xgboost import plot_importance


def load_data():
    df = pd.read_csv(
        "./datasets/ratings.csv",
        sep=",",
        encoding = "latin",
        low_memory = False
    )
    data = pd.DataFrame(df)
    id = [i for i in range(len(df))]
    data['id'] = id
    return df

book_rating_df = load_data()[1:]
book_rating_df.duplicated(["book_id","user_id","rating"]).sum()

split_value = int(len(book_rating_df) * 0.80)
train_data = book_rating_df[:split_value]
test_data = book_rating_df[split_value:]

# -----------------------------------------------------------------------
# showing number of each ratings
# -----------------------------------------------------------------------
# plt.figure(figsize = (12, 8))
# ax = sns.countplot(x="rating", data=train_data)

# ax.set_yticklabels([num for num in ax.get_yticks()])

# plt.tick_params(labelsize = 15)
# plt.title("Count Ratings in train data", fontsize = 20)
# plt.xlabel("Ratings", fontsize = 20)
# plt.ylabel("Number of Ratings", fontsize = 20)
# plt.show()


# -----------------------------------------------------------------------
# matrix user/item/rating
# -----------------------------------------------------------------------
def get_user_item_sparse_matrix(df):
    sparse_data = sparse.csr_matrix( (df.rating, (df.user_id, df.book_id )) )
    return sparse_data

train_sparse_data = get_user_item_sparse_matrix(train_data)
test_sparse_data = get_user_item_sparse_matrix(test_data)
global_average_rating = train_sparse_data.sum() / train_sparse_data.count_nonzero()


# -----------------------------------------------------------------------
# get average rating for user/book
# -----------------------------------------------------------------------
def get_average_rating(sparse_matrix, is_user):
    ax = 1 if is_user else 0
    sum_of_ratings = sparse_matrix.sum(axis = ax).A1
    no_of_ratings = (sparse_matrix != 0).sum(axis = ax).A1
    rows, cols = sparse_matrix.shape
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i] for i in range(rows if is_user else cols) if no_of_ratings[i] != 0}
    return average_ratings

average_rating_user = get_average_rating(train_sparse_data, True)       #avg of rating that each user rate 
avg_rating_book = get_average_rating(train_sparse_data, False)          #avg of rating for each book


# -----------------------------------------------------------------------
# extra information about our train dataset
# -----------------------------------------------------------------------
total_users = len(np.unique(book_rating_df["user_id"]))
train_users = len(average_rating_user)
uncommonUsers = total_users - train_users
# print("Total no. of Users = {}".format(total_users))
# print("No. of Users in train data= {}".format(train_users))
# print("No. of Users not present in train data = {}({}%)".format(uncommonUsers, np.round((uncommonUsers/total_users)*100), 2))

total_book = len(np.unique(book_rating_df["book_id"]))
train_book = len(avg_rating_book)
uncommonBooks = total_book - train_book
# print("Total no. of Books = {}".format(total_book))
# print("No. of Books in train data= {}".format(train_book))
# print("No. of Books not present in train data = {}({}%)".format(uncommonBooks, np.round((uncommonBooks/total_book)*100), 2))


# -----------------------------------------------------------------------
# read books.csv
# -----------------------------------------------------------------------
book_titles_df = pd.read_csv("./datasets/books.csv",
                                sep=",",
                                encoding="latin",
                                low_memory=False)

Book_DataFrame = pd.DataFrame(book_titles_df)


# -----------------------------------------------------------------------
# compute similarity of other books with specified book
# -----------------------------------------------------------------------
def compute_book_similarity_count(sparse_matrix, book_titles_df, book_id):
    similarity = cosine_similarity(sparse_matrix.T, dense_output = False)
    no_of_similar_books = book_titles_df.loc[book_id]['original_title'], similarity[book_id].toarray()
    return no_of_similar_books


# -----------------------------------------------------------------------
# suggest books to new user
# -----------------------------------------------------------------------

# first way(bad way):
# book_data_sort = book_rating_df.sort_values(by=['rating'], ascending=False)
# print(book_data_sort.head(10))

# second way(suggested way):
def suggest(read_book_ids):
    all_suggested = []
    all_name = []
    removed = []
    for book_id in read_book_ids:
        name, similar_books = compute_book_similarity_count(train_sparse_data, book_titles_df, book_id)
        all_name.append(name)
        similar_books_id = [(similar_books[0][i], i) for i in range(len(similar_books[0]))]
        final_similar_books = []
        for book in similar_books_id:
            final_similar_books.append((book[0],Book_DataFrame.iloc[book[1]]['original_title']))
        final_similar_books = sorted(final_similar_books, key=lambda x: x[0])[-10:]
        all_suggested.extend(final_similar_books)
    all_suggested = sorted(all_suggested, key=lambda x: x[0])
    for tup in all_suggested:
        if tup[1] in all_name:
            removed.append(tup)
    for tup in removed:
        all_suggested.remove(tup)
    print(all_name)
    for book in all_suggested[-10:]:
        print(book[1], book[0])

suggest([1])


# -----------------------------------------------------------------------
# featureing way
# -----------------------------------------------------------------------
def get_sample_sparse_matrix(sparse_matrix, no_of_users, no_of_books):
    users, books, ratings = sparse.find(sparse_matrix)
    uniq_users = np.unique(users)
    uniq_books = np.unique(books)
    np.random.seed(15)
    user = np.random.choice(uniq_users, no_of_users, replace = False)
    book = np.random.choice(uniq_books, no_of_books, replace = True)
    mask = np.logical_and(np.isin(users, user), np.isin(books, book))
    sparse_matrix = sparse.csr_matrix((ratings[mask], (users[mask], books[mask])),
                                                     shape = (max(user)+1, max(book)+1))
    return sparse_matrix

train_sample_sparse_matrix = get_sample_sparse_matrix(train_sparse_data, 400, 40)       #(user id, book id) rating
test_sparse_matrix_matrix = get_sample_sparse_matrix(test_sparse_data, 200, 20)


def create_new_similar_features(sample_sparse_matrix):
    global_avg_rating = get_average_rating(sample_sparse_matrix, False)
    global_avg_users = get_average_rating(sample_sparse_matrix, True)
    global_avg_books = get_average_rating(sample_sparse_matrix, False)
    sample_train_users, sample_train_books, sample_train_ratings = sparse.find(sample_sparse_matrix)
    new_features_csv_file = open("./datasets/new_features.csv", mode = "w")

    for user, book, rating in zip(sample_train_users, sample_train_books, sample_train_ratings):
        similar_arr = list()
        similar_arr.append(user)
        similar_arr.append(book)
        similar_arr.append(sample_sparse_matrix.sum()/sample_sparse_matrix.count_nonzero())

        similar_users = cosine_similarity(sample_sparse_matrix[user], sample_sparse_matrix).ravel()
        indices = np.argsort(-similar_users)[1:]
        ratings = sample_sparse_matrix[indices, book].toarray().ravel()
        top_similar_user_ratings = list(ratings[:5])
        top_similar_user_ratings.extend([global_avg_rating[book]] * (5 - len(ratings)))
        similar_arr.extend(top_similar_user_ratings)

        similar_books = cosine_similarity(sample_sparse_matrix[:,book].T, sample_sparse_matrix.T).ravel()
        similar_books_indices = np.argsort(-similar_books)[1:]
        similar_books_ratings = sample_sparse_matrix[user, similar_books_indices].toarray().ravel()
        top_similar_book_ratings = list(similar_books_ratings[:5])
        top_similar_book_ratings.extend([global_avg_users[user]] * (5-len(top_similar_book_ratings)))
        similar_arr.extend(top_similar_book_ratings)

        similar_arr.append(global_avg_users[user])
        similar_arr.append(global_avg_books[book])
        similar_arr.append(rating)

        new_features_csv_file.write(",".join(map(str, similar_arr)))
        new_features_csv_file.write("\n")

    new_features_csv_file.close()
    new_features_df = pd.read_csv('./datasets/new_features.csv', names = ["user_id", "book_id", "gloabl_average", "similar_user_rating1",
                                                               "similar_user_rating2", "similar_user_rating3",
                                                               "similar_user_rating4", "similar_user_rating5",
                                                               "similar_book_rating1", "similar_book_rating2",
                                                               "similar_book_rating3", "similar_book_rating4",
                                                               "similar_book_rating5", "user_average",
                                                               "book_average", "rating"])
    return new_features_df

train_new_similar_features = create_new_similar_features(train_sample_sparse_matrix)
train_new_similar_features = train_new_similar_features.fillna(0)
test_new_similar_features = create_new_similar_features(test_sparse_matrix_matrix)
test_new_similar_features = test_new_similar_features.fillna(0)
x_train = train_new_similar_features.drop(["user_id", "book_id", "rating"], axis = 1)
x_test = test_new_similar_features.drop(["user_id", "book_id", "rating"], axis = 1)
y_train = train_new_similar_features["rating"]
y_test = test_new_similar_features["rating"]

def error_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

clf = XGBRegressor(n_estimators = 45, silent = False, n_jobs  = 10, objective="reg:squarederror")
clf.fit(x_train, y_train)

y_pred_test = clf.predict(x_test)
# for i in range(len(y_test)):
#     print(y_pred_test[i], y_test[i])
rmse_test = error_metrics(y_test, y_pred_test)
# print("RMSE = {}".format(rmse_test))        #Root mean squared error
