import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        self.user_top_items = data.groupby('user_id').apply(lambda x: 
                                                            [k for k, v in sorted([(row, qt) for row, qt in zip(x['item_id'], x['quantity']) if row != 999999], 
                                                                                  key=lambda y: y[1], 
                                                                                  reverse=True)])
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data, values='quantity', aggfunc='count'):
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values=values, 
                                  aggfunc=aggfunc, 
                                  fill_value=0
                                 )
        user_item_matrix = user_item_matrix.astype(float) #
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        top_items = self.user_top_items[user]
        recs_for_items = []
        for item in top_items[:N]:
            recs_for_items.append([self.id_to_itemid[self.model.similar_items(self.itemid_to_id[item], N=N + 1)[i][0]] for i in range(1, N + 1)])
        
        # Стараемся добиться максимально релевантных, неповторяющихся рекомендаций
        rec_list = []
        for i in range(1, N+1):
            for recs in recs_for_items:
                for rec in recs[:i]:
                    if rec not in rec_list:
                        rec_list.append(rec)
            if len(rec_list) >= N:
                rec_list = rec_list[:N]
                break

        assert len(rec_list) == N, 'Количество рекомендаций != {}'.format(N)
        return rec_list
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        res = self.own_recommender.recommend(userid=self.userid_to_id[user], 
                        user_items=csr_matrix(self.user_item_matrix).tocsr(), 
                        N=N, 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
