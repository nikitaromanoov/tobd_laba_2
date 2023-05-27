import pickle
import numpy as np
import pandas as pd
import json
from sklearn import tree

import os
import redis

class ModelPrediction:
    
    def __init__(self, path_dataset):
        with open("./src/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        df_test = pd.read_csv("./data/test.csv")
        
    def predict(self, X):
        return self.model.predict(X)
        
def redis_f(name, value):
        
        r = redis.Redis(host=os.environ.get("REDIS_ADDRESS"),
                        port=int(os.environ.get("REDIS_PORT")),
                        username=os.environ.get("REDIS_USER"),
                        password=os.environ.get("REDIS_PASSWORD"),
                        decode_responses=True)

        r.set(name, value)

        return r.get(name)      
        

def main():

	trainer =  ModelPrediction("./data")
	with open("./tests/test_0.json") as f:
	    d = json.load(f)
	    predictions = trainer.predict(d["X"])
	    predictions = redis_f("prediction 0", str(predictions) )
	    print(predictions, d["y"])
	with open("./tests/test_1.json") as f:
	    d = json.load(f)
	    predictions = trainer.predict(d["X"])
	    predictions = redis_f("prediction 1", str(predictions) )
	    print(predictions, d["y"])	

if __name__ == '__main__':
	main()
