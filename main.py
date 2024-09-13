import pandas as pd
import pycaret
from pycaret.classification import *

data= pd.read_csv(r"./data/raw_data.csv")
data.head()


s=setup(
      data, target = 'Churn', session_id = 123,
      ignore_features = ['customerID'],
      log_experiment = True, 
      experiment_name = 'customer_churn',
      )

best = compare_models()

# plot_model(best, plot = 'auc')
# plot_model(best, plot = 'confusion_matrix')
# plot_model(best, plot = 'boundary')
# plot_model(best, plot = 'learning')


save_model(best, './outputs/model')