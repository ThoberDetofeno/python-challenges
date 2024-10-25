# You are given a set of variants with 5 distinct activities [a, e] (see below for more details).
# We want to calculate the minimum set of variants, so that each activity is at least present once.
# We are adding the constraint to prefer the most frequent variants.
# Example:
#   <a,b,c,d>100
#   <a,b,e>50
#   <d,e,c>10
# [[0, 100, 'a', 'b', 'c', 'd'], [1, 50, ] ]
# Given the minimal example above, variants <a,b,c,d> and <a,b,e> would be selected to cover all activities at least once. 
# Variant <d,e,c> is ignored as all activities are already being present in the previous, more frequent variants.
# The given text file has the following format and consists of 10 distinct variants:
# <frequency>, <comma separated activities/traces/variants>
# Input data (also available in text file variants):
#     100, a, b
#     95, a, c
#     95, c, d, d
#     90, c, d, d, e
#     90, a, c, c, e
#     85, a, b, c, d
#     85, d
#     80, e, c
#     80, b, c, e
#     80, c, e

import itertools
import os

# List of activities valid
activities = ['a', 'b', 'c', 'd', 'e']


def calculate(variants_data: list) -> list:
    """ Calculate of variants
        
    Calculate the minimum set of variants with most frequent variants.
    The function calculate the minimum set of variants with 5 distinct activities [a,e], adding the constraint to prefer the most frequent variants.
    The function has 3 main steps: 
        * First step is a loop with the number of possible combinations, starting with 1 up to 5 which is the maximum number of activities.
        * Second step are generated the "n" variants combination
        * Third step it is checked whether the combination has all activities and is calculated then the most frequency variants.

    Parameters
    ----------
    variants_data : list
        A variant list is acceptable with the format is [[variant 1],[variant 2],...,[variant n]]

    Returns
    -------
    list
       Minimum set of variants with format [variant_index_1, variant_index_2,...,variant_index_n]

    Example
    -------
    >>> variants = calculate(variants_data)

    """
    # List of combination [1, 2, 3, 4, 5]
    combinations = list(range(1, len(activities) + 1))
    variants = []
    max = -1
    # For each "n" combination
    for nr_combination in combinations:
        # Generate "n" variants combination
        for var_combination in itertools.combinations(variants_data, nr_combination):
            # Join the activities of variants combination
            var_activity = ''.join([''.join(var[2:]) for var in var_combination])
            # Identify set of variants with 5 distinct activities [a,e]
            if all(act in var_activity for act in activities):
                sum_var = sum([int(var[1]) for var in var_combination[0 : nr_combination]])
                # Discovery the most frequent variants
                if (sum_var > max):
                    max = sum_var
                    variants = [var[0] for var in var_combination[0 : nr_combination]] 
        # Verify if exists a set of variants
        if (len(variants)):
            return variants
    # No set variants found
    return variants


def read_data_file(file_path: str) -> list:
    """Import data file

    Import file into Python list and validate the datas.
    The text file must be the following format:
        <frequency>, <comma separated activities/traces/variants>
    Data file example:
        100, a, b
        95, a, c
        95, c, d, d

    Parameters
    ----------
    file_path : str
        A data file path valid.

    Returns
    -------
    list
        Variant list with format [[variant 1], [variant 2], ..., [variant n]].
        Variant format [<index>, <frequency>, <comma separated activities/traces/variants>]
        Variant list example:  [[0, 100, 'a', 'b'], [1, 95,'a', 'c'], [2, 95,'c', 'd', 'd']]

    Example
    -------
    >>> data_log = read_data_file('C:\\celonis_prj\\data\\variants_')
    
    """
    # Delimiter used in data file
    separator = ','
    
    variant_data = []
    index = 0
    # Open data file
    file_data = open(file_path)
    
    while True:
        content = file_data.readline()
        if not content:
            break
        content = content.replace('\n', '')
        # Validate variant data. 
        content_list = str(index) + separator + content.replace(' ', '')
        content_list = content_list.split(separator)
        # Validate If the two firsts element are number and other list elements are an activity valid [a-e]
        if (all([act_value.isdigit() for act_value in content_list[0:2]])  and 
                all([act in activities for act in content_list[2:]])):
            print('    ' + content)
            variant_data.append(content_list)
            index += 1
        else:
            print('    ' + content + ' --> ERROR')
    
    file_data.close()
    return variant_data


def select_data_file() -> str:
    """Select data file

    Select data file of "data" directory.
    The "data" directory must be inside current working directory.

    Returns
    -------
    str
        A data file path

    Example
    -------
    >>> file_path = select_data_file()

    """
    file_list = []
    dir_path = os.getcwd()+"\\code_challenge\\data"
    # Create a data file list
    for file_name in os.listdir(dir_path):
        # Check if current path is a file
        if os.path.isfile(os.path.join(dir_path, file_name)):
            file_list.append(dir_path+'\\'+file_name)
    
    # Return the data file selected
    match len(file_list):
        case 0:
            print("Add the files in the folder (../data).")
            return ""
        case 1:
            print("Data file: " + file_list[0])
            return file_list[0]
        case _:
            # Show the list of data file
            print("List of data file.")
            print("Number | Name")
            print("--------------------------------------")
            for file_name in file_list:
                print(str(file_list.index(file_name) + 1) +'      | ' + file_name)
            print("--------------------------------------")
            # Select the data file
            select_file = input('Enter the file number: ')
            file_number = int(select_file) - 1 if select_file.isdigit() else -1
            # Validate If exists the data file and return the path
            if file_number in list(range(len(file_list))):
                print("Data file: " + file_list[file_number])
                return file_list[file_number]
            else: 
                return ""


def main():
    """Main function

    Python main function is a starting point of program.
    Main function is executed only when it is run as a Python program. 
    It will not run the main function if it imported as a module.

    """
    while True:
        file_path = select_data_file()
        if file_path == "":
            print("\nNo file selected.")
            return
        else:
            variant_data = read_data_file(file_path)
            variants_index = calculate(variant_data)
            if (len(variants_index)):
                print("Set of variant:")
                for index in variants_index:
                    print('    ' + ', '.join(map(str, variant_data[int(index)][1:])))
            else:
                print("\nNo variant selected.")
        
        if (input('Do you want to Continue (yes/no)? ') == 'no'):
            print("By!\n")
            return


if __name__ == "__main__":
    main()
