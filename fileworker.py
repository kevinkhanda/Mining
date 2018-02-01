import numpy as np
import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori

data = []

with open('DUMP.txt', encoding='utf-16') as content:
    i = -1
    j = 0
    for line in content:
        line = line[0:-1]
        if line == '**SOF**':
            i += 1
            data.append([])
        else:
            if line != '**EOF**':
                data[i].append(line)

transactions = OnehotTransactions()
fitted = transactions.fit(data).transform(data)
dataframe = pd.DataFrame(fitted, columns=transactions.columns_)
print(dataframe)

print(apriori(dataframe, min_support=0.1, use_colnames=True))
