import math
import time
from pytrends.request import TrendReq


def generate_data(index_, iterations_, kw_list_, no_dups_, pytrend_):
    # pytrend = TrendReq()
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
            for row in inner_list.get(entry).get('top').get('query'):
                if row not in no_dups_:
                    no_dups_.add(row)
                    new_list.append(row)
    generate_data(index_, iterations_, new_list, no_dups_, pytrend_)


kw_list = ['COVID19', 'corona']
no_dups = set()
index = 0
iterations = 2
pytrend = TrendReq(hl='en-US', tz=360, timeout=(10, 25), proxies=['https://192.168.0.106:80'], retries=5,
                   backoff_factor=1)

generate_data(index, iterations, kw_list, no_dups, pytrend)

print('######')
print(len(no_dups))
print(no_dups)

# interest_by_region_df = pytrend.interest_by_region()
# print(interest_by_region_df)
# print(interest_by_region_df.loc[ 'United States' , : ])

# historical_interest_df =pytrend.get_historical_interest(['COVID19'], year_start=2020, month_start=3, day_start=20, hour_start=0, year_end=2020, month_end=3, day_end=21, hour_end=0, geo='US-PA', sleep=60)
# print(historical_interest_df)
