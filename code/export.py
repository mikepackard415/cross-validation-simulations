import numpy as np
import pandas as pd

# Export results
def exp(sims, tables, sizes, procedures, models, D, loc):

    columns=["Model 1","Model 2","Model 3"]
    for i in range(models-3):
        j=str(i+4)
        columns.append("Model " + j)
    
    dgp_col = []
    for t in tables:
        dgp_col.extend([t[0]] * len(sizes) * len(procedures))
    
    sizes_col=[]
    for i in sizes:
        #j = str(i)
        for _ in procedures:
            sizes_col.append(i)
    sizes_col=sizes_col * len(tables)
    
    procs_col = procedures * len(tables) * len(sizes)

    sims_col = [sims] * len(tables) * len(sizes) * len(procedures)

    finaldf = pd.DataFrame(np.reshape(D,(-1, models)), dgp_col, columns)
    finaldf.insert(0, 'Sample Size', sizes_col)
    finaldf.insert(1, 'Procedure',   procs_col)
    finaldf.insert(2, 'Iterations',  sims_col)

    if loc == '':
        pd.set_option('display.max_rows', None)
        print(finaldf)
    else:
        finaldf.to_excel(loc)
