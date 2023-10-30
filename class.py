import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNWithZScore

class RecommendationSystem:
    def _init_(self, user_data_list):
        self.user_data = pd.DataFrame(user_data_list, columns=['user_id', 'product_id', 'rating'])
        self.reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(self.user_data[['user_id', 'product_id', 'rating'], self.reader])
        self.trainset, self.testset = train_test_split(self.data, test_size=0.25)
        self.model = KNNWithZScore(sim_options={'name': 'cosine', 'user_based': False})
        self.model.fit(self.trainset)

    def get_recommendations(self, user_id, num_recommendations=10):
        user_products = self.user_data[self.user_data['user_id'] == user_id]['product_id']
        unrated_products = self.user_data[~self.user_data['product_id'].isin(user_products)]
        user_predictions = [self.model.predict(user_id, product_id) for product_id in unrated_products['product_id']]
        top_n = sorted(user_predictions, key=lambda x: x.est, reverse=True)
        recommendations = [{'Product ID': prediction.iid, 'Estimated Rating': prediction.est} for prediction in top_n[:num_recommendations]]
        return recommendations

# Ejemplo de uso
user_data_list = [
    [1, 101, 5],
    [1, 102, 4],
    [1, 103, 3],
    [2, 104, 4],
    [2, 105, 5]
]

recommendation_system = RecommendationSystem(user_data_list)
user_id = 1
recommendations = recommendation_system.get_recommendations(user_id)
print('Recomendaciones para el usuario:')
for recommendation in recommendations:
    print(f'Producto ID: {recommendation["Product ID"]}, Calificación Estimada: {recommendation["Estimated Rating"]}')