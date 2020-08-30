import zenitai
from zenitai.transform.woe import WoeTransformer, WoeTransformerRegularized

wt = WoeTransformer()
print(wt)

wt = WoeTransformerRegularized()
print(wt)
