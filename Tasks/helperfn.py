import pandas
import os.path

def _get_random_data(filepath):
    """Use get_random_data() instead. 
    Get Dataframe with randomized instance order, takes filepath as arg

    :param filepath: [description]
    :type filepath: [type]
    :return: [description]
    :rtype: [type]
    """
    df = pandas.read_csv(filepath, header='infer')
    return df.sample(frac=1)


def _get_file_path(filename):
    """Construct the filepath string, takes name of file as arg

    :param filename: [description]
    :type filename: [type]
    :return: [description]
    :rtype: [type]
    """
    my_path = os.path.abspath(os.path.dirname("Data/"))
    return os.path.join(my_path, filename)

def get_random_data(filename):
    """Get Dataframe with randomized instance order, takes filename as arg

    :param filename: [description]
    :type filename: [type]
    :return: [description]
    :rtype: [type]
    """
    return _get_random_data(_get_file_path('x_train_gr_smpl.csv'))

