import pandas as pd
from tools.analysis.eval_top5_stroke_properties import df_res

# Print clean table
print(df_res.to_markdown(index=False))
