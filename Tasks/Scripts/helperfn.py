import pandas
import numpy as np
import os.path

from Scripts import pixelFinder



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

label_def = {
    -1: 'All Classes',
    0: 'speed limit 20',
    1: 'speed limit 30',
    2: 'speed limit 50',
    3: 'speed limit 60',
    4: 'speed limit 70',
    5: 'left turn',
    6: 'right turn',
    7: 'beware pedestrian crossing',
    8: 'beware children',
    9: 'beware cycle route ahead'
}



def get_pixel_class_tuple(pixels, label):
    """[summary]

    Args:
        pixels ([type]): [description]
        label ([type]): [description]

    Returns:
        [type]: A list of n pixel tuples with the best
    """
    pixelsResult = pixelFinder.bestPixels(label, pixels)    
    convertedLabel = label_def[1]
    
    return [(str(pixel), convertedLabel) for pixel in pixelsResult]

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

    :param data: The dataset without the result column
    :type data: pandas.df
    :param result: The result vector to append to data
    :type result: pandas.df
    """
    result.columns = ['y']
    return data.join(result)

def randomize_data(dataframe):
    """dumb randomize, no discretization

    :param dataframe: [description]
    :type dataframe: [type]
    """
    return dataframe.sample(frac=1)

def balance_by_class(X, y, size=None, allow_imbalance=False):
    """Select a sample of the data with a balanced class distribution

    :param X: data
    :type X: pandas.df
    :param y: labels
    :type y: pandas.df
    :param samples: size of sample. Defaults to None -> in this case the sample returned will be the size of the smallest class
    :type samples: int, Optional
    :param allow_imbalance: If size param > number of rows in smallest class indicate if allowing an imbalanced class distribution is ok
    :type allow_imbalance: Bool, Optional
    :return: the sample and labels
    :rtype: tuple(pandas.df, pandas.df)
    """
    data = append_result_col(X, y)
    classes = np.unique(y)
    datasets = []
    smallest_size = data.shape[0]
    for i in range(len(classes)):
        temp_sample = data[data['y'] == classes[i]]
        datasets += [temp_sample.sample(frac=1)]
        if temp_sample.shape[0] < smallest_size:
            smallest_size = temp_sample.shape[0]
    frame = pandas.DataFrame(columns=data.columns)
    if size is None: 
        for df in datasets:
            frame = frame.append(df.head(smallest_size))
    else:
        if allow_imbalance and size > smallest_size:
            raise ValueError(
                "Size argument is too large for a balanced dataset")
        for df in datasets:
            frame = frame.append(df.head(size))
    y_res = frame[['y']]
    X_res = frame.drop('y', 1)
    return X_res.astype(int), y_res.astype(int)


    