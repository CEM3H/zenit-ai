# тесты для монотонных границ

# %%
import sys
sys.path.insert(0, r'C:\Users\s.kadochnikov\Desktop\Проекты\zenitai-lib')
import pandas as pd
import numpy as np
import woeTransformer_beta as woe

# %%
def poly_test(df_list):
    return [np.polyfit(i['predictor'], i['target'], deg=1)]

def group_test(df_list):
    return [woe.grouping(i['predictor'], i['target'], deg=1)]

def monotonic_test(df_list, p_list)


# %%
test_df = pd.DataFrame({'predictor':np.random.choice(range(10), size=100),
                        'target':np.random.choice([0,1], size=100)})

test_df2 = test_df.copy()
test_df2 = test_df2.append(pd.DataFrame({'predictor':['-1']*10,
                                         'target':np.random.choice([0,1], size=10)}))

test_df3 = test_df.copy()
test_df3 = test_df3.append(pd.DataFrame({'predictor':['text_value']*10,
                                         'target':np.random.choice([0,1], size=10)}))

test_df4 = test_df.copy()
test_df4 = test_df4.append(pd.DataFrame({'predictor':np.array([np.nan]*10),
                                         'target':np.random.choice([0,1], size=10)})
                            )
# %%
p = np.polyfit(test_df4['predictor'],
               test_df4['target'], 
               deg=1)
print(p)
# %%
grouped = woe.grouping(test_df4)
grouped

# %%
monotonic = woe.monotonic_borders(grouped, p[::-1])
monotonic

# %%
