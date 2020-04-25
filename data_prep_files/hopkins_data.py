import csv
import os


def get_state_info(path_):
    with open(path_, encoding="utf-8") as file:
        state_names_and_abbreviations = []
        for line in file:
            name_and_abbreviation = [x.strip() for x in line.split(',')]
            temp = (name_and_abbreviation[0], name_and_abbreviation[1])
            state_names_and_abbreviations.append(temp)
    return state_names_and_abbreviations


def clean_format_1_data(path_, output_path_):
    just_one_loop = False
    state_and_abbreviations_path = './states_and_abbreviations.txt'
    states = get_state_info(state_and_abbreviations_path)
    for filename in os.listdir(path_):
        day, month, year = filename.split('.')[0].split('-')
        new_filename = "{}-{}-{}.csv".format(year, day, month)
        for state in states:
            confirmed = 0
            deaths = 0
            recovered = 0
            with open(path_ + filename, encoding="utf-8") as file:
                for values in csv.reader(file):
                    if state[0] in values[0] or state[1] in values[0]:
                        if values[3].strip() != '':
                            confirmed += int(values[3])
                        if values[4].strip() != '':
                            deaths += int(values[4])
                        if values[5].strip() != '':
                            recovered += int(values[5])


            f = open(cleaned_path + new_filename, "a")
            f.write(str(confirmed) + ',' + str(deaths) + ',' + str(recovered) + '\n')
            f.close()
            if just_one_loop:
                return


def clean_format_2_data(path_, output_path_):
    just_one_loop = False
    state_and_abbreviations_path = './states_and_abbreviations.txt'
    states = get_state_info(state_and_abbreviations_path)
    for filename in os.listdir(path_):
        day, month, year = filename.split('.')[0].split('-')
        new_filename = "{}-{}-{}.csv".format(year, day, month)
        for state in states:
            confirmed = 0
            deaths = 0
            recovered = 0
            with open(path_ + filename, encoding="utf-8") as file:
                for values in csv.reader(file):
                    if state[0] in values[2] or state[1] in values[2]:
                        if values[7].strip() != '':
                            confirmed += int(values[7])
                        if values[8].strip() != '':
                            deaths += int(values[8])
                        if values[9].strip() != '':
                            recovered += int(values[9])

            f = open(cleaned_path + new_filename, "a")
            f.write(str(confirmed) + ',' + str(deaths) + ',' + str(recovered) + '\n')
            f.close()
            if just_one_loop:
                return


format_1_path = './hopkins_data/format_1/'
format_2_path = './hopkins_data/format_2/'
# cleaned_path = './hopkins_data/cleaned/'
cleaned_path = './raw/y/'
clean_format_1_data(format_1_path, cleaned_path)
clean_format_2_data(format_2_path, cleaned_path)
