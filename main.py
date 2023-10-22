import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.prediction_algorithms.knns import KNNWithZScore
 
# DataFrame con datos de Amazon (Reemplazar datos)
data = pd.DataFrame({​​​​​​
    'user_id': [1, 2, 3, 4, 5],
    'product_id': [101, 102, 103, 104, 105],
    'category': ['Electronics', 'Books', 'Products', 'Cleaning', 'Clothing'],
    'price': [100, 20, 120, 25, 50],
    'rating': [5, 4, 3, 4, 5],
    'reviews': [10, 5, 12, 7, 8]  
}​​​​​​)
 
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)
 
model = KNNWithZScore(sim_options={​​​​​​'name': 'cosine', 'user_based': False}​​​​​​)
model.fit(trainset)
 
predictions = model.test(testset)
 
rmse = accuracy.rmse(predictions)
print(f'RMSE: {​​​​​​rmse}​​​​​​')
 
user_id = 1
user_products = data[data['user_id'] == user_id]['product_id']
unrated_products = data[~data['product_id'].isin(user_products)]
user_predictions = [model.predict(user_id, product_id) for product_id in unrated_products['product_id']]
 
top_n = sorted(user_predictions, key=lambda x: x.est, reverse=True)
 
# Mostramos recomendaciones
print(f'Recomendaciones para el usuario {​​​​​​user_id}​​​​​​:')
for prediction in top_n[:10]:
    print(f'Producto ID: {​​​​​​prediction.iid}​​​​​​, Calificación Estimada: {​​​​​​prediction.est}​​​​​​')
