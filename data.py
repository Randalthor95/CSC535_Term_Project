import math
from pytrends.request import TrendReq
from datetime import timedelta, date

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)



def generate_search_terms(index_, iterations_, kw_list_, no_dups_, pytrend_):
    if index_ >= iterations_:
        return

    print(index_)
    print(kw_list_)
    index_ += 1

    new_lists = []
    for x in range(math.ceil(len(kw_list_) / 5)):
        pytrend_.build_payload(kw_list=kw_list_[x:x + 5])
        related = pytrend_.related_queries()
        new_lists.append(related)

    new_list = []
    for inner_list in new_lists:
        for entry in inner_list:
            get_entry = inner_list.get(entry)
            get_top = get_entry.get('top')
            if get_top is not None:
                get_query = get_top.get('query')
                for row in get_query:
                    if row not in no_dups_:
                        no_dups_.add(row)
                        new_list.append(row)
    generate_search_terms(index_, iterations_, new_list, no_dups_, pytrend_)


# timeframe=2016-12-14 2017-01-25
def generate_state_level_data(kw_list_, start_date, end_date):
    previous_date = ''
    date_time_format_string = "%Y-%m-%d"
    df = pd.DataFrame()
    for current_date in daterange(start_date, end_date):
        if previous_date == '':
            previous_date = current_date
            continue
        print(current_date)
        timeframe_ = previous_date.strftime(date_time_format_string) + ' ' + current_date.strftime(date_time_format_string)
        interest_list = []
        for entry in kw_list_:
            new_kw_list = [entry]
            pytrend.build_payload(kw_list=new_kw_list, geo='US', timeframe=timeframe_)
            interest_by_region_df = pytrend.interest_by_region(resolution='Region')
            interest_list.append(interest_by_region_df)
        df[previous_date.strftime(date_time_format_string)] = interest_list
        previous_date = current_date
    return df


kw_list = ['COVID19', 'corona', 'cough', 'fever', 'coronavirus symptoms']
no_dups = set()
index = 0
iterations = 4
pytrend = TrendReq(hl='en-US', tz=360, timeout=(10, 25), proxies=['https://192.168.0.107:80'], retries=5,
                   backoff_factor=.1)

# generate_search_terms(index, iterations, kw_list, no_dups, pytrend)
# print(no_dups)
# print('######')

state_data = generate_state_level_data(kw_list, date(2020, 2, 14), date(2020, 2, 15))
print(state_data['2020-02-14'].to_list())