import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings('ignore',category='DeprecationWarning')
#warnings.filterwarnings('ignore',category='UserWarning')
sns.set_style('whitegrid')
np.random.seed(7)
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances



def preprocess(feature):
  def lemmatize_text(feature):
    return[lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(feature)]
    feature = feature.apply(lemmatize_text)
    feature = feature.apply(lambda x:" ".join(x))

  transformed_feature=tfidf_vector.transform(feature)
  #transformed_feature = tfidf_vector.transform(feature)

  return transformed_feature 




