from . import helperfn as hf
from . import pixelFinder as pf
import os
import pandas as pd

def _get_file_path(filename):
    """Construct the filepath string, takes name of file as arg

    :param filename: The name of the file to be imported
    :type filename: string
    :return: The absolute path to the file
    :rtype: string
    """
    my_path = os.path.abspath(os.path.dirname("Data/weka_files/"))
    return os.path.join(my_path, filename)

def wekaPixelChooser(input_file, output_file, best_pixels_num, positive_num, negative_num):
    """Picks desired pixels and rows from weka file into a new weka file for Bayes Net analysis

    :param input_file: The name of the file to be imported
    :type input_file: string
    :param output_file: The name of the file to be exported to
    :type output_file: string
    :param best_pixels_num: Number of best pixels to use
    :type best_pixels_num: int
    :param positive_num: Number of positive rows to use
    :type positive_num: int
    :param negative_num: Number of negative rows to use
    :type negative_num: int
    """
    #Input list of pixel indexes, from Task 4
    pixels = pf.bestPixels(0, best_pixels_num)

    # add attributes to top of export file
    f = open(_get_file_path(output_file), "w")

    attributes_txt = "@relation x_train_gr_smpl\n"

    for i in pixels:
        attributes_txt += "@attribute X" + str(i) + " numeric \n"

    attributes_txt += "@attribute LABEL {True, False}\n@Data\n"

    f.write(attributes_txt)
    f.close()

    #input weka file but only the data so it looks like csv
    df = pd.read_csv (_get_file_path(input_file), header=None)

    #group by outcome, two groups 0 and 1
    grouped = df.groupby(2304, as_index = False)

    #get dataframe of each group
    g0 = grouped.get_group(0)
    g1 = grouped.get_group(1)

    #get n random rows from group 0 (CHOOSE HOW MANY POSITIVE ROWS YOU WANT)
    random_g0 = g0.sample(positive_num, random_state=20)
    # random_g0

    # get m random rows from group 1 (CHOOSE HOW MANY NEGATIVE ROWS YOU WANT)
    random_g1 = g1.sample(negative_num, random_state=20)
    # random_g1


    #get only the defined columns and outcome which is column 2304, columns == list of pixel indexes
    #concat them together as a single DataFrame
    choosen_indexes = pd.concat([random_g0[pixels + [2304]], random_g1[pixels + [2304]]])
    # choosen_indexes

    #replace 0 and 1 outcomes to True and False respectively
    choosen_indexes[2304].replace([0, 1], ['True', 'False'], inplace=True)
    choosen_indexes

    # export weka values to new file
    choosen_indexes.to_csv(_get_file_path(output_file), header=None, index=None, mode='a')
