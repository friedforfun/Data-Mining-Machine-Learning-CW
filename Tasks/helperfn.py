import pandas
import os.path

def _get_random_data(filepath):
    """Use get_random_data() instead. 
    Get Dataframe with randomized instance order, takes filepath as arg

    :param filepath: The full absolute path to the file
    :type filepath: string
    :return: The entire data collection with randomized order of instances
    :rtype: pandas datafram
    """
    df = pandas.read_csv(filepath, header='infer')
    return df.sample(frac=1)


def _get_file_path(filename):
    """Construct the filepath string, takes name of file as arg

    :param filename: The name of the file to be imported
    :type filename: string
    :return: The absolute path to the file
    :rtype: string
    """
    my_path = os.path.abspath(os.path.dirname("Data/"))
    return os.path.join(my_path, filename)

# use this function to get the full csv
def get_random_data(filename):
    """Get Dataframe with randomized instance order, takes filename as arg

    :param filename: The name of the file to be imported
    :type filename: string
    :return: The entire data collection with randomized order of instances
    :rtype: pandas dataframe
    """
    return _get_random_data(_get_file_path(filename))

