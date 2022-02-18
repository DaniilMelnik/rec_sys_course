import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000, item_features=None):
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    
    # уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    data = data[data['price'] > 1]
    # уберем слишком дорогие товары
    data = data[data['price'] < 60]
    
    # уберем не интересные дл¤ рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

   # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['share_unique_users'] = popularity['user_id'] / data['user_id'].nunique()

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # уберем товары, которые не продавались за последние 12 мес¤цев
    week_of_last_purchaices = data.groupby('item_id')['week_no'].max().reset_index()
    bought_year_ago = week_of_last_purchaices[week_of_last_purchaices['week_no'] < (data['week_no'].max() - 52)].item_id.tolist()
    data = data[~data['item_id'].isin(bought_year_ago)]
    
    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    return data
    
def postfilter_items(user_id, recommednations):
    pass