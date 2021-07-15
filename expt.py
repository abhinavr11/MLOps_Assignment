import pandas as pd 
import dvc.api
x = dvc.api.get_url(repo ="https://github.com/abhinavr11/MLOps_Assignment", path = "data/creditcard.csv" )
print(x)

with dvc.api.open(repo ="https://github.com/abhinavr11/MLOps_Assignment", path = "data/creditcard.csv" , mode = "r" ) as fd:
    df = pd.read_csv(fd)

