import csv
import math
import time
import random

import pytrends
import requests
from fake_useragent import UserAgent

from pytrends.request import TrendReq
from datetime import timedelta, date

import pandas as pd
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
    previous_date = ''
    df = pd.DataFrame()
    tor = launch_tor_with_config(tor_cmd=tor_path, init_msg_handler=print_lines, config={'ControlPort': '9051'})
    headers = {'User-Agent': UserAgent().random}
    proxies = {
        'http': 'socks5://127.0.0.1:9050',
        'https': 'socks5://127.0.0.1:9050'
    }
    c = Controller.from_port(port=9051)
    c.authenticate('tor')
    c.signal(Signal.NEWNYM)
    print(requests.get('http://icanhazip.com', proxies=proxies, headers=headers).content)

    pytrend = TrendReq(hl='en-US', tz=360, timeout=(10, 25), proxies=['https://127.0.0.1:9050'], retries=5,
                       backoff_factor=1)
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
                print(count)
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
                            time.sleep(random.randint(60, 600))
                            c = Controller.from_port(port=9051)
                            c.authenticate('tor')
                            c.signal(Signal.NEWNYM)
                            print(requests.get('http://icanhazip.com', proxies=proxies, headers=headers).content)
                            pytrend = TrendReq(hl='en-US', tz=360, timeout=(5, 120), proxies=['https://127.0.0.1:9050'],
                                               retries=5,
                                               backoff_factor=1)
                            pytrend.build_payload(kw_list=new_kw_list, geo='US', timeframe=timeframe_)
                            interest_by_region_df = pytrend.interest_by_region(resolution='Region')
                            interest_list.append(interest_by_region_df)
                        except (pytrends.request.exceptions.ResponseError, requests.exceptions.ReadTimeout) as inner_e:
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


def generate_state_level_data_proxies(kw_list_, start_date, end_date, proxies_):
    previous_date = ''
    df = pd.DataFrame()

    pytrend = TrendReq(hl='en-US', tz=360, timeout=(10, 25), proxies=proxies_, retries=5,
                       backoff_factor=1)

    for current_date in daterange(start_date, end_date):
        print(current_date)
        if previous_date == '':
            previous_date = current_date
            continue
        timeframe_ = previous_date.strftime(date_time_format_string) + ' ' + current_date.strftime(
            date_time_format_string)
        interest_list = []
        count = 1
        for entry in kw_list_:
            if count % 25 == 0:
                print(count)
            new_kw_list = [entry]
            pytrend.build_payload(kw_list=new_kw_list, geo='US', timeframe=timeframe_)
            interest_by_region_df = pytrend.interest_by_region(resolution='Region')
            interest_list.append(interest_by_region_df)
            count += 1
        df[previous_date.strftime(date_time_format_string)] = interest_list
        previous_date = current_date

    return df


def save_dates_data_to_csv(path_, start_date_, end_date_, dates_data):
    for current_date in daterange(start_date_, end_date_):
        if current_date == end_date_:
            continue
        current_date_as_string = current_date.strftime(date_time_format_string)
        dates_data[current_date_as_string].to_csv(path_ + '//' + current_date_as_string + '.csv')


path = "C:\\Users\\Randalthor95\\Documents\\cs535\\"

# index = 0
# iterations = 7
# date1 = date(2020, 1, 31)
# date2 = date(2020, 2, 1)
# pytrend = TrendReq(hl='en-US', tz=360, timeout=(10, 25), proxies=['https://127.0.0.1:9050'], retries=5,
#                    backoff_factor=1)
# kw_list = ['COVID19']
# timeframe_ = date1.strftime(date_time_format_string) + ' ' + date2.strftime(
#             date_time_format_string)
# pytrend.build_payload(kw_list=kw_list, geo='US', timeframe=timeframe_)
# interest_by_region_df = pytrend.interest_by_region(resolution='Region')

# no_dups = set()
# generate_search_terms(index, iterations, kw_list, no_dups, pytrend)
# save_search_terms_to_csv(path + "search_terms.csv", no_dups)

# proxies = ['https://albany.cs.colostate.edu:60021',
#            'https://richmond.cs.colostate.edu:60021',
#            'https://sacramento.cs.colostate.edu:60021',
#            'https://saint-paul.cs.colostate.edu:60021',
#            'https://salem.cs.colostate.edu:60021',
#            'https://salt-lake-city.cs.colostate.edu:60021',
#            'https://santa-fe.cs.colostate.edu:60021',
#            'https://springfield.cs.colostate.edu:60021',
#            'https://tallahassee.cs.colostate.edu:60021',
#            'https://earth.cs.colostate.edu:60021']

proxies = [
    'https://annapolis.cs.colostate.edu:60021',
    'https://atlanta.cs.colostate.edu:60021',
    'https://augusta.cs.colostate.edu:60021',
    'https://austin.cs.colostate.edu:60021',
    'https://baton-rouge.cs.colostate.edu:60021',
    'https://bismarck.cs.colostate.edu:60021',
    'https://boise.cs.colostate.edu:60021',
    'https://boston.cs.colostate.edu:60021',
    'https://carson-city.cs.colostate.edu:60021',
    'https://charleston.cs.colostate.edu:60021',
    'https://columbia.cs.colostate.edu:60021',
    'https://columbus-oh.cs.colostate.edu:60021',
    'https://concord.cs.colostate.edu:60021',
    'https://denver.cs.colostate.edu:60021',
    'https://des-moines.cs.colostate.edu:60021',
    'https://dover.cs.colostate.edu:60021',
    'https://frankfort.cs.colostate.edu:60021',
    'https://harrisburg.cs.colostate.edu:60021',
    'https://hartford.cs.colostate.edu:60021',
    'https://helena.cs.colostate.edu:60021',
    'https://honolulu.cs.colostate.edu:60021',
    'https://indianapolis.cs.colostate.edu:60021',
    'https://jackson.cs.colostate.edu:60021',
    'https://jefferson-city.cs.colostate.edu:60021',
    'https://juneau.cs.colostate.edu:60021',
    'https://lansing.cs.colostate.edu:60021',
    'https://lincoln.cs.colostate.edu:60021',
    'https://madison.cs.colostate.edu:60021',
    'https://montgomery.cs.colostate.edu:60021',
    'https://montpelier.cs.colostate.edu:60021',
    'https://nashville.cs.colostate.edu:60021',
    'https://oklahoma-city.cs.colostate.edu:60021',
    'https://olympia.cs.colostate.edu:60021',
    'https://phoenix.cs.colostate.edu:60021',
    'https://pierre.cs.colostate.edu:60021',
    'https://providence.cs.colostate.edu:60021',
    'https://raleigh.cs.colostate.edu:60021',
    'https://richmond.cs.colostate.edu:60021',
    'https://sacramento.cs.colostate.edu:60021',
    'https://saint-paul.cs.colostate.edu:60021',
    'https://salem.cs.colostate.edu:60021',
    'https://salt-lake-city.cs.colostate.edu:60021',
    'https://santa-fe.cs.colostate.edu:60021',
    'https://springfield.cs.colostate.edu:60021',
    'https://tallahassee.cs.colostate.edu:60021',
    'https://topeka.cs.colostate.edu:60021',
    'https://trenton.cs.colostate.edu:60021',
    'https://ankara.cs.colostate.edu:60021',
    'https://baghdad.cs.colostate.edu:60021',
    'https://bangkok.cs.colostate.edu:60021',
    'https://beijing.cs.colostate.edu:60021',
    'https://berlin.cs.colostate.edu:60021',
    'https://bogota.cs.colostate.edu:60021',
    'https://cairo.cs.colostate.edu:60021',
    'https://dhaka.cs.colostate.edu:60021',
    'https://hanoi.cs.colostate.edu:60021',
    'https://hong-kong.cs.colostate.edu:60021',
    'https://jakarta.cs.colostate.edu:60021',
    'https://kabul.cs.colostate.edu:60021',
    'https://kinshasa.cs.colostate.edu:60021',
    'https://lima.cs.colostate.edu:60021',
    'https://london.cs.colostate.edu:60021',
    'https://madrid.cs.colostate.edu:60021',
    'https://mexico-city.cs.colostate.edu:60021',
    'https://moscow.cs.colostate.edu:60021',
    'https://pyongyang.cs.colostate.edu:60021',
    'https://riyadh.cs.colostate.edu:60021',
    'https://santiago.cs.colostate.edu:60021',
    'https://seoul.cs.colostate.edu:60021',
    'https://singapore.cs.colostate.edu:60021',
    'https://tehran.cs.colostate.edu:60021',
    'https://tokyo.cs.colostate.edu:60021',
    'https://anchovy.cs.colostate.edu:60021',
    'https://barracuda.cs.colostate.edu:60021',
    'https://blowfish.cs.colostate.edu:60021',
    'https://bonito.cs.colostate.edu:60021',
    'https://brill.cs.colostate.edu:60021',
    'https://bullhead.cs.colostate.edu:60021',
    'https://char.cs.colostate.edu:60021',
    'https://cod.cs.colostate.edu:60021',
    'https://dorado.cs.colostate.edu:60021',
    'https://eel.cs.colostate.edu:60021',
    'https://flounder.cs.colostate.edu:60021',
    'https://grouper.cs.colostate.edu:60021',
    'https://halibut.cs.colostate.edu:60021',
    'https://herring.cs.colostate.edu:60021',
    'https://mackerel.cs.colostate.edu:60021',
    'https://marlin.cs.colostate.edu:60021',
    'https://perch.cs.colostate.edu:60021',
    'https://pollock.cs.colostate.edu:60021',
    'https://sardine.cs.colostate.edu:60021',
    'https://shark.cs.colostate.edu:60021',
    'https://sole.cs.colostate.edu:60021',
    'https://swordfish.cs.colostate.edu:60021',
    'https://tarpon.cs.colostate.edu:60021',
    'https://turbot.cs.colostate.edu:60021',
    'https://tuna.cs.colostate.edu:60021',
    'https://wahoo.cs.colostate.edu:60021',
    'https://a-basin.cs.colostate.edu:60021',
    'https://ajax.cs.colostate.edu:60021',
    'https://beaver-creek.cs.colostate.edu:60021',
    'https://breckenridge.cs.colostate.edu:60021',
    'https://buttermilk.cs.colostate.edu:60021',
    'https://cooper.cs.colostate.edu:60021',
    'https://copper-mtn.cs.colostate.edu:60021',
    'https://crested-butte.cs.colostate.edu:60021',
    'https://eldora.cs.colostate.edu:60021',
    'https://grandby-ranch.cs.colostate.edu:60021',
    'https://aspen-highlands.cs.colostate.edu:60021',
    'https://howelsen-hill.cs.colostate.edu:60021',
    'https://keystone.cs.colostate.edu:60021',
    'https://loveland.cs.colostate.edu:60021',
    'https://mary-jane.cs.colostate.edu:60021',
    'https://monarch.cs.colostate.edu:60021',
    'https://powderhorn.cs.colostate.edu:60021',
    'https://purgatory.cs.colostate.edu:60021',
    'https://silverton.cs.colostate.edu:60021',
    'https://snowmass.cs.colostate.edu:60021',
    'https://steamboat.cs.colostate.edu:60021',
    'https://sunlight.cs.colostate.edu:60021',
    'https://vail.cs.colostate.edu:60021',
    'https://winter-park.cs.colostate.edu:60021',
    'https://wolf-creek.cs.colostate.edu:60021',
    'https://earth.cs.colostate.edu:60021',
    'https://jupiter.cs.colostate.edu:60021',
    'https://mars.cs.colostate.edu:60021',
    'https://mercury.cs.colostate.edu:60021',
    'https://neptune.cs.colostate.edu:60021',
    'https://saturn.cs.colostate.edu:60021',
    'https://uranus.cs.colostate.edu:60021',
    'https://venus.cs.colostate.edu:60021',
    'https://bentley.cs.colostate.edu:60021',
    'https://bugatti.cs.colostate.edu:60021',
    'https://ferrari.cs.colostate.edu:60021',
    'https://jaguar.cs.colostate.edu:60021',
    'https://lamborghini.cs.colostate.edu:60021',
    'https://lotus.cs.colostate.edu:60021',
    'https://maserati.cs.colostate.edu:60021',
    'https://porsche.cs.colostate.edu:60021',
    'https://washington-dc.cs.colostate.edu:60021',
    'https://acorn.cs.colostate.edu:60021',
    'https://ginko.cs.colostate.edu:60021',
    'https://heartnut.cs.colostate.edu:60021',
    'https://nangai.cs.colostate.edu:60021',
    'https://pili.cs.colostate.edu:60021',
    'https://pinion.cs.colostate.edu:60021',
    'https://pistachio.cs.colostate.edu:60021',
    'https://bananas.cs.colostate.edu:60021',
    'https://raspberries.cs.colostate.edu:60021',
    'https://pomegranates.cs.colostate.edu:60021',
    'https://dates.cs.colostate.edu:60021',
    'https://eggplant.cs.colostate.edu:60021',
    'https://endive.cs.colostate.edu:60021',
    'https://fennel.cs.colostate.edu:60021',
    'https://garlic.cs.colostate.edu:60021',
    'https://gourd.cs.colostate.edu:60021',
    'https://horseradish.cs.colostate.edu:60021',
    'https://kale.cs.colostate.edu:60021',
    'https://kelp.cs.colostate.edu:60021',
    'https://leek.cs.colostate.edu:60021',
    'https://lettuce.cs.colostate.edu:60021',
    'https://mushroom.cs.colostate.edu:60021',
    'https://okra.cs.colostate.edu:60021',
    'https://onion.cs.colostate.edu:60021',
    'https://parsley.cs.colostate.edu:60021',
    'https://parsnip.cs.colostate.edu:60021',
    'https://pea.cs.colostate.edu:60021',
    'https://pepper.cs.colostate.edu:60021',
    'https://potato.cs.colostate.edu:60021',
    'https://pumpkin.cs.colostate.edu:60021',
    'https://radish.cs.colostate.edu:60021',
    'https://rhubarb.cs.colostate.edu:60021',
    'https://romanesco.cs.colostate.edu:60021',
    'https://rutabaga.cs.colostate.edu:60021',
    'https://shallot.cs.colostate.edu:60021',
    'https://squash.cs.colostate.edu:60021',
    'https://tomatillo.cs.colostate.edu:60021',
    'https://tomato.cs.colostate.edu:60021',
    'https://turnip.cs.colostate.edu:60021',
    'https://wasabi.cs.colostate.edu:60021',
    'https://yam.cs.colostate.edu:60021',
    'https://zucchini.cs.colostate.edu:60021',
    'https://uncompahgre.cs.colostate.edu:60021',
    'https://kiwis.cs.colostate.edu:60021',
    'https://nectarines.cs.colostate.edu:60021',
    'https://peaches.cs.colostate.edu:60021',
    'https://coopersmiths.cs.colostate.edu:60021',
    'https://dc-oakes.cs.colostate.edu:60021',
    'https://equinox.cs.colostate.edu:60021',
    'https://funkwerks.cs.colostate.edu:60021',
    'https://maxline.cs.colostate.edu:60021',
    'https://new-belgium.cs.colostate.edu:60021',
    'https://odell.cs.colostate.edu:60021',
    'https://rally-king.cs.colostate.edu:60021',
    'https://uno.cs.colostate.edu:60021',
    'https://acushla.cs.colostate.edu:60021'
]

kw_list = read_terms_from_csv("./search_terms.csv")
# kw_list = ['COVID19', 'corona', 'cough', 'fever', 'coronavirus symptoms']
for i in range(7, 1, -1):
    start_date = date(2020, 4, i-1)
    end_date = date(2020, 4, i)
    start_time = time.time()
    state_data = generate_state_level_data_proxies(kw_list, start_date, end_date, proxies)
    print("--- %s seconds ---" % (time.time() - start_time))
    save_dates_data_to_csv("dates", start_date, end_date, state_data)
    print("--- %.2gs seconds ---" % (time.time() - start_time))
    time.sleep(random.randrange(3600, 5400))

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
