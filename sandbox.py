"""
Модуль для тестирования
"""


from zenitai.transform import WoeTransformer
from zenitai.utils.utils import generate_train_data

df = generate_train_data()

wt = WoeTransformer()
# print(wt)
