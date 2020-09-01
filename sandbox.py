# import zenitai
# import matplotlib.pyplot as plt
# from zenitai.transform.woe import WoeTransformer, WoeTransformerRegularized
from zenitai.utils.utils import generate_train_data
from zenitai.transform import WoeTransformer, grouping, statistic, cat_features_alpha_logloss, woeTransformer

df = generate_train_data()


woeTransformer(df['digits'], df['target'])
wt = WoeTransformer()
# print(wt)
