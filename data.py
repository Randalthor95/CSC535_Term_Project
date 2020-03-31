import csv
import math
import time
import random

import pytrends
import requests

from pytrends.request import TrendReq
from datetime import timedelta, date

import pandas as pd
import socks
import socket
from stem import Signal
from stem.control import Controller
from stem.process import launch_tor_with_config

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def generate_search_terms(index_, iterations_, kw_list_, no_dups_, pytrend_):
    if index_ >= iterations_:
        return

    print(index_)
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


def save_search_terms_to_csv(path_, search_terms_set):
    with open(path_, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(list(search_terms_set))


date_time_format_string = "%Y-%m-%d"


def read_terms_from_csv(path_):
    with open(path_, encoding="utf-8") as file:
        reader = csv.reader(file)
        return list(reader)[0]


tor_path = "C:\\Users\\Randalthor95\\Desktop\\Tor Browser\\Browser\\TorBrowser\\Tor\\tor.exe"


def print_lines(line):
    if ('Bootstrapped' in line):
        print(line)



# timeframe=2016-12-14 2017-01-25
def generate_state_level_data(kw_list_, start_date, end_date):
    pytrend = TrendReq(hl='en-US', tz=360)
    previous_date = ''
    df = pd.DataFrame()
    tor = launch_tor_with_config(tor_cmd=tor_path, init_msg_handler=print_lines, config={'ControlPort': '9051'})
    proxies = {
        'http': 'socks5://127.0.0.1:9050',
        'https': 'socks5://127.0.0.1:9050'
    }
    c = Controller.from_port(port=9051)
    c.authenticate('tor')
    c.signal(Signal.NEWNYM)
    print(requests.get('http://icanhazip.com', proxies=proxies).content)

    for current_date in daterange(start_date, end_date):
        if previous_date == '':
            previous_date = current_date
            continue
        timeframe_ = previous_date.strftime(date_time_format_string) + ' ' + current_date.strftime(
            date_time_format_string)
        interest_list = []
        count = 1
        for entry in kw_list_:
            try:
                new_kw_list = [entry]
                pytrend.build_payload(kw_list=new_kw_list, geo='US', timeframe=timeframe_)
                interest_by_region_df = pytrend.interest_by_region(resolution='Region')
                interest_list.append(interest_by_region_df)
            except pytrends.request.exceptions.ResponseError as e:
                try:
                    print(count)
                    print(e)
                    while True:
                        try:
                            time.sleep(random.randint(10, 20))
                            c = Controller.from_port(port=9051)
                            c.authenticate('tor')
                            c.signal(Signal.NEWNYM)
                            print(requests.get('http://icanhazip.com', proxies=proxies).content)
                            pytrend = TrendReq(hl='en-US', tz=360)
                            pytrend.build_payload(kw_list=new_kw_list, geo='US', timeframe=timeframe_)
                            interest_by_region_df = pytrend.interest_by_region(resolution='Region')
                            interest_list.append(interest_by_region_df)
                        except pytrends.request.exceptions.ResponseError as inner_e:
                            print('Try again...')
                        except Exception:
                            raise
                        else:
                            break
                except Exception as e:
                    print(e)
                    tor.terminate()
                    raise
            except Exception as e:
                print(count)
                print(type(e))
                print(e)
                tor.terminate()
                raise
            count += 1

        df[previous_date.strftime(date_time_format_string)] = interest_list
        previous_date = current_date

    tor.terminate()

    return df


def save_dates_data_to_csv(path_, start_date_, end_date_, dates_data):
    for current_date in daterange(start_date_, end_date_):
        if current_date == end_date_:
            continue
        current_date_as_string = current_date.strftime(date_time_format_string)
        dates_data[current_date_as_string].to_csv(path_ + '\\' + current_date_as_string + '.csv')


path = "C:\\Users\\Randalthor95\\Documents\\cs535\\"

# index = 0
# iterations = 7
# pytrend = TrendReq(hl='en-US', tz=360, timeout=(10, 25), proxies=['https://127.0.0.1:9050'], retries=5,
#                    backoff_factor=1)
# kw_list = ['COVID19', 'corona', 'cough', 'fever', 'coronavirus symptoms']
# no_dups = set()
# generate_search_terms(index, iterations, kw_list, no_dups, pytrend)
# save_search_terms_to_csv(path + "search_terms.csv", no_dups)

kw_list = read_terms_from_csv(path + "search_terms.csv")
start_date = date(2020, 2, 14)
end_date = date(2020, 2, 15)
start_time = time.time()
state_data = generate_state_level_data(kw_list, start_date, end_date)
print("--- %s seconds ---" % (time.time() - start_time))
save_dates_data_to_csv(path + "dates", start_date, end_date, state_data)
print("--- %.2gs seconds ---" % (time.time() - start_time))

# tor = launch_tor_with_config(tor_cmd=tor_path, init_msg_handler=print_lines, config={'ControlPort': '9051'})
# proxies = {
#     'http': 'socks5://127.0.0.1:9050',
#     'https': 'socks5://127.0.0.1:9050'
# }

# with Controller.from_port(port=9051) as c:
#     c.authenticate('tor')
#     c.signal(Signal.NEWNYM)
#     old_proxy = requests.get('http://icanhazip.com', proxies=proxies).content.decode("utf-8")
#     new_proxy = old_proxy
#     print(new_proxy)
# with Controller.from_port(port=9051) as c:
#     time.sleep(10)
#     c.authenticate('tor')
#     c.signal(Signal.NEWNYM)
#
#     new_proxy = requests.get('http://icanhazip.com', proxies=proxies).content.decode("utf-8")
#     print(new_proxy)
#     tor.terminate()