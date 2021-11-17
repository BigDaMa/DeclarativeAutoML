import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml

dictionary = pickle.load(open('/tmp/constraints_training_time.p', 'rb'))

print(dictionary)

dictionary = pickle.load(open('/tmp/constraints_inference_time.p', 'rb'))

print(dictionary)


dictionary = pickle.load(open('/tmp/constraints_pipeline_size.p', 'rb'))

print(dictionary)