from pytrends.request import TrendReq
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)


def generate_data(index_, iterations_, kw_list_, no_dups_):

    if index_ >= iterations_:
        return
    print(index_)
    index_ += 1

    pytrend.build_payload(kw_list=kw_list_)
    related = pytrend.related_queries()
    new_list = []
    for entry in kw_list_:
        for row in related.get(entry).get('top').get('query'):
            no_dups_.add(row)
            new_list.append(row)
    print(new_list)
    generate_data(index_, iterations_, new_list, no_dups_)

pytrend = TrendReq()

# Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()

kw_list = ['COVID19', 'corona']

no_dups = set()

# print(no_dups)

index = 0
iterations = 2

generate_data(index, iterations, kw_list, no_dups)

print('######')
print(len(no_dups))
print(no_dups)

# interest_by_region_df = pytrend.interest_by_region()
# print(interest_by_region_df)
# print(interest_by_region_df.loc[ 'United States' , : ])

# historical_interest_df =pytrend.get_historical_interest(['COVID19'], year_start=2020, month_start=3, day_start=20, hour_start=0, year_end=2020, month_end=3, day_end=21, hour_end=0, geo='US-PA', sleep=60)
# print(historical_interest_df)
