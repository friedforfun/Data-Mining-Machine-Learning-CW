import pandas
import os.path

def _get_dataset(filepath):
    """Use get_random_data() instead. 
    Get Dataframe with randomized instance order, takes filepath as arg

    :param filepath: The full absolute path to the file
    :type filepath: string
    :return: The entire data collection with randomized order of instances
    :rtype: pandas datafram
    """
    return pandas.read_csv(filepath, header='infer')



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
def get_random_data(result_id=-1):
    """Get Dataframe with randomized instance order, takes filename as arg

    :param filename: The name of the file to be imported
    :type filename: string
    :return: A tuple of entire data collection with randomized order of instances
    :rtype: (pandas.df, pandas.df)
    """
    result_files = {
        -1: 'y_train_smple.csv',
        0: 'y_train_smple_0.csv',
        1: 'y_train_smple_1.csv',
        2: 'y_train_smple_2.csv',
        3: 'y_train_smple_3.csv',
        4: 'y_train_smple_4.csv',
        5: 'y_train_smple_5.csv',
        6: 'y_train_smple_6.csv',
        7: 'y_train_smple_7.csv',
        8: 'y_train_smple_8.csv',
        9: 'y_train_smple_9.csv'
    }
    data = _get_dataset(_get_file_path('x_train_gr_smpl.csv'))
    filePicker = result_files.get(result_id, 'y_train_smple.csv')
    result = _get_dataset(_get_file_path(filePicker))
    full_data = randomize_data(append_result_col(data, result))
    y = full_data[['y']]
    x = full_data.drop('y', 1)
    return (x, y)

def append_result_col(data, result):
    """Return a new dataframe with result column in data

    :param data: [description]
    :type data: [type]
    :param result: [description]
    :type result: [type]
    """
    result.columns = ['y']
    return data.assign(result)

def randomize_data(dataframe):
    """[summary]

    :param dataframe: [description]
    :type dataframe: [type]
    """
    return dataframe.sample(frac=1)
