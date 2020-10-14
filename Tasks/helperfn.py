import pandas
import os.path

result_files = {
    -1: 'y_train_smpl.csv',
    0: 'y_train_smpl_0.csv',
    1: 'y_train_smpl_1.csv',
    2: 'y_train_smpl_2.csv',
    3: 'y_train_smpl_3.csv',
    4: 'y_train_smpl_4.csv',
    5: 'y_train_smpl_5.csv',
    6: 'y_train_smpl_6.csv',
    7: 'y_train_smpl_7.csv',
    8: 'y_train_smpl_8.csv',
    9: 'y_train_smpl_9.csv'
}

def _file_selector(id):
    return result_files.get(id, 'y_train_smpl.csv')

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

    :return: A tuple of entire data collection with randomized order of instances
    :rtype: (pandas.df, pandas.df)
    """
    x, y = get_data(result_id)
    full_data = randomize_data(append_result_col(x, y))
    y = full_data[['y']]
    x = full_data.drop('y', 1)
    return x, y


def get_data(result_id=-1):
    """Get a tuple of the result and data csv

    :param result_id: The index of the result datafile, defaults to -1
    :type result_id: int, optional
    :return: A tuple of the data collection
    :rtype: (pandas.df, pandas.df)
    """
    x = _get_dataset(_get_file_path('x_train_gr_smpl.csv'))
    filePicker = result_files.get(result_id, 'y_train_smpl.csv')
    y = _get_dataset(_get_file_path(filePicker))
    y.columns = ['y']
    return x, y

def get_results(result_id=-1):
    """Get the result dataset on its own

    :param result_id: The id of the result file, -1 indicates the full classifier, defaults to -1
    :type result_id: int, optional
    :return: classification results in order
    :rtype: pandas.df
    """
    return _get_dataset(_get_file_path(_file_selector(result_id)))

def get_data_noresults():
    """Get the dataset without results

    :return: The raw dataset, in order, without result column
    :rtype: pandas.df
    """
    return _get_dataset(_get_file_path('x_train_gr_smpl.csv'))


def append_result_col(data, result):
    """Return a new dataframe with result column in data

    :param data: [description]
    :type data: [type]
    :param result: [description]
    :type result: [type]
    """
    result.columns = ['y']
    return data.join(result)

def randomize_data(dataframe):
    """[summary]

    :param dataframe: [description]
    :type dataframe: [type]
    """
    return dataframe.sample(frac=1)
