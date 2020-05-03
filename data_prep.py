#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys
import pandas as pd
import numpy as np
import re

files = []
for line in sys.stdin:
    files.append(line.strip())

# print(files)

for file in files:
    print(file)
    df = pd.read_csv(file)
    df.columns = ['term_index', 'search_term']
    # 51 states by 5 search terms
    data_matrix = np.empty((51, len(df['search_term'][:])))
    term_number = 0
    for df_row in df['search_term'][:]:
        search_term_field = df_row.split('\n')
        search_term = search_term_field[0].strip()
        states_scores = search_term_field[2:]
        state_number=0
        for line in states_scores:
            splits = re.split(r'\s{2,}', line)
            state, score = splits[0], splits[1]
            data_matrix[state_number][term_number] = score
            state_number = state_number + 1
        term_number = term_number + 1
    save_file_to = 'datesprepped/' + file.split('/')[1]
    np.savetxt(save_file_to, data_matrix, fmt="%d")
    print('Saved to', save_file_to)