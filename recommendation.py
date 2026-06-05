import pandas as pd
from normalize import hybrid_search

# class to store a query into a user profile embedding
class Recommender:
    def __init__(self):
        # we need a list to store the last 10 queries searched by the use
        self.history = []

    def log_query(self, query):
        self.history.append(query)
        if len(self.history) > 10:
            self.history.pop(0)

    def generate_recommendations(self):
        if not self.history:
            return pd.DataFrame()
        user_query = ' '.join(self.history)
        recommendations = hybrid_search(user_query, b=0.75)

        return recommendations[['docno', 'hybrid_score']].head(10)
    
rec_engine = Recommender()

rec_engine.log_query("Healthy hot snack")
rec_engine.log_query("peanut free snack")

print(rec_engine.generate_recommendations())