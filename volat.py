# The function below takes the file name corresponding to a stock and returns the stock_id

def stock_id(name):
    import re
    i = re.search(r'\d+', name).group(0)
    i = int(i)
    return i

# The function below takes the path of a folder containing stock files and returns a list of the contained file paths
# sorted by stock_id


def files(folder):
    import glob

    file_names = glob.glob(folder+'/*')
    file_names.sort(key=stock_id)
    return file_names

# The function below takes a list of stock files, a number n and and a boolean value. And returns the concatenation of
# of n the respective dataframes with and additional 'stock_id' columns. If the boolean value is TRUE it uses the first
# n files, if the boolean value is FALSE it chooses randomly n files.


def sub_frame(file_names, number, sequential):
    import random
    import pandas as pd

    if sequential:
        file_names = file_names[0:number]
    else:
        file_names = random.sample(file_names, number)

    data_frames = []
    for filename in file_names:
        i = stock_id(filename)
        parquet_file = str(filename)
        frame = pd.read_parquet(parquet_file, engine='auto')
        size = frame.shape[0]
        column = [i] * size
        frame['ID'] = column
        data_frames.append(frame)

    data = pd.concat(data_frames)
    return data

# The function below takes a stock dataframe and returns the first and last row of a random group. By group we mean a
# a set of rows corresponding to the same time_id. The probability of each group is analogous to its size.


def random_batch(frame):
    import random
    n = frame.shape[0]
    stock_list = list(range(n))

    k = random.choice(stock_list)
    i = k
    right_end = k
    while frame.loc[i].at["time_id"] == frame.loc[k].at["time_id"]:
        right_end = i
        i = i + 1
    i = k
    left_end = k
    while frame.loc[i].at["time_id"] == frame.loc[k].at["time_id"]:
        left_end = i
        i = i - 1

    batch_locations = list(range(left_end, right_end + 1))
    return batch_locations


# The function below prints a parquet file given it's path.


def parquet_print(filename):
    import pandas as pd
    parquet_file = str(filename)
    frame = pd.read_parquet(parquet_file, engine='auto')
    print(frame)

# Similarly for csv files


def csv_print(filename):
    import pandas as pd
    parquet_file = str(filename)
    frame = pd.read_csv(parquet_file)
    print(frame)


# The function below returns the dataframe contained in a parquet file given its path.

def parquet_frame(filename):
    import pandas as pd
    parquet_file = str(filename)
    frame = pd.read_parquet(parquet_file)
    return frame

#Similarly for csv files


def csv_frame(filename):
    import pandas as pd
    parquet_file = str(filename)
    frame = pd.read_csv(parquet_file)
    return frame

# The function below takes a stock data frame and a number k and returns the first and last row of the
# time_id-group containing the k_th row


def local_patch(frame, k):
    n = frame.shape[0]
    time_id = frame.loc[k].at["time_id"]

    if time_id == 5:
        i = k
        right_end = k
        while frame.loc[i].at["time_id"] == time_id:
            right_end = i
            i = i + 1
        patch_locations = (time_id, 0, right_end)
    elif time_id == 32767:
        i = k
        left_end = k
        while frame.loc[i].at["time_id"] == time_id:
            left_end = i
            i = i - 1
        patch_locations = (time_id, left_end, n-1)
    else:
        i = k
        right_end = k
        while frame.loc[i].at["time_id"] == time_id:
            right_end = i
            i = i + 1
        i = k
        left_end = k
        while frame.loc[i].at["time_id"] == time_id:
            left_end = i
            i = i - 1
        patch_locations = (time_id, left_end, right_end)
    return patch_locations


# The function below takes in a stock dataframe and a number between r 0 and 1. It returns a random list
# of time_id-groups so that the total arrows of the groups do not exceed r*(size of the dataframe). The time_id groups
# are represented as (first row of group, last row fo group)

def file_sample(frame, percentage):
    import random

    n = frame.shape[0]
    rows = 0
    patches = []
    k = random.choice(range(n))
    patch = local_patch(frame, k)
    rows = rows + patch[2] - patch[1] + 1
    while rows <= percentage * n:
        patches.append(patch)
        k = random.choice(range(n))
        patch = local_patch(frame, k)
        rows = rows + patch[2] - patch[1] + 1
    return patches


# The function below takes in a stock dataframe a list of time_id-groups of that dataframe (groups represented as in
# the function above) and a real number r. It splits the list of time_id groups according to the real number and then
# concatenates  the data frames corresponding to it's part returning two dataframes.


def patches_to_frame(frame, patches, train_size):
    import math
    import pandas as pd

    n = len(patches)
    k = math.floor(n*train_size)
    patches1 = patches[0:k]
    patches2 = patches[k:n]
    patches = [patches1, patches2]

    frames = []
    for j in range(2):
        data_frames = []
        for i in patches[j]:
            data_frames.append(frame[i[1]:i[2]+1])
        data = pd.concat(data_frames)
        frames.append(data)

    return [frames[0], frames[1]]

# The function below take a list of pats of stock files a real number p and a real number t. It randomly chooses time_id
# groups from each file corresponding approximately to r*100 percentage of total rows in the files, also it splits
# the list of groups according to t keeping a list of training groups and a list of test groups. Finally it concatenates
# all the groups from all the files returning two data frames corresponding to a training dataframe and a test
# dataframe.


def global_random_sample(file_names, percentage, train_size):
    import pandas as pd

    train_frames = []
    test_frames = []
    for filename in file_names:
        i = stock_id(filename)
        frame = parquet_frame(filename)
        patches = file_sample(frame, percentage)
        frames = patches_to_frame(frame, patches, train_size)

        size = frames[0].shape[0]
        column = [i] * size
        frames[0]['stock_id'] = column
        train_frames.append(frames[0])

        size = frames[1].shape[0]
        column = [i] * size
        frames[1]['stock_id'] = column
        test_frames.append(frames[1])

    data1 = pd.concat(train_frames)
    data2 = pd.concat(test_frames)
    return [data1, data2]

# The following function returns a 3D as required for the keras LSTM model. We pad each time_id group with 0s
# so that the size of all groups is equal to 'groupsize'. The 'first' and 'last arguments corresponding to the first
# and last column of 'frame' that will be used as values for the LSTM model.


def lstm_input(frame, groupsize, first, last):
    import numpy as np
    import pandas as pd

    n = frame.shape[0]

    previous = 0
    inpt = np.array([np.zeros((groupsize, last-first))])
    for i in range(n - 1):
        if not frame.loc[i].at["time_id"] == frame.loc[i + 1].at["time_id"]:
            matrix = pd.DataFrame.to_numpy(frame.iloc[previous:i + 1, first:last])
            pad = [[0] * (last-first)] * (groupsize - i - 1 + previous)
            matrix = np.concatenate((pad, matrix), axis=0)
            inpt = np.append(inpt, [matrix], axis=0)
            previous = i + 1
    matrix = pd.DataFrame.to_numpy(frame.iloc[previous:n, first:last])
    pad = [[0] * (last-first)] * (groupsize - n + previous)
    matrix = np.concatenate((pad, matrix), axis=0)
    inpt = np.append(inpt, [matrix], axis=0)
    inpt = np.delete(inpt, 0, axis=0)
    return inpt

# The following function takes a dataframe that is a concatenation of stock dataframes and returns a subframe containing
# all rows corresponding to stock_id = sid. It is required that the initial frame has a stock_id column.


def stock_subframe(frame, sid):
    import numpy as np

    temp = np.array(frame['stock_id'])
    temp2 = np.where(temp == sid)[0]
    start = temp2[0]
    end = start + len(temp2) - 1
    subframe = frame.loc[start: end]
    return subframe

# The following function takes a dataframe of concatenated stock dataframes and returns the list of target values
# of the corresponding (stock_id,time_id) elements.


def frame_to_values(frame, values_fr):
    import numpy as np

    m = values_fr.shape[0]

    y = []
    for i in range(m):
        if i == 0:
            sid = values_fr.loc[i].at["stock_id"]
            subframe = stock_subframe(frame, sid)
            sample_ids = subframe.loc[:, "time_id"]
            sample_ids = set(sample_ids)
        elif not values_fr.loc[i].at["stock_id"] == values_fr.loc[i-1].at["stock_id"]:
            sid = values_fr.loc[i].at["stock_id"]
            subframe = stock_subframe(frame, sid)
            sample_ids = subframe.loc[:, "time_id"]
            sample_ids = set(sample_ids)

        if values_fr.loc[i].at["time_id"] in sample_ids:
            y.append(values_fr.loc[i].at["target"])
    y = np.array(y)
    return y

# In the following function filename1 corresponds to the path of a file that is a random sample of some book-stock_id
# file and filename2 is the path of the corresponding trade-stock_id file. It returns the random sample of filename2
# that contains the time_id-groups as filename2


def counterpart_file(filename1, filename2):
    import pandas as pd

    frame2 = parquet_frame(filename2)
    frame1 = csv_frame(filename1)
    sid = stock_id(filename2)
    subframe = stock_subframe(frame1, sid)
    sample_ids = subframe.loc[:, "time_id"]
    sample_ids = set(sample_ids)

    frames = []
    previous = 0
    n = frame2.shape[0]
    for i in range(n):
        if i == n - 1:
            if frame2.loc[i].at["time_id"] in sample_ids:
                subframe = frame2.loc[previous + 1:n - 1]
                frames.append(subframe)
            previous = i
        elif not frame2.loc[i + 1].at["time_id"] == frame2.loc[i].at["time_id"]:
            if frame2.loc[i].at["time_id"] in sample_ids:
                subframe = frame2.loc[previous + 1:i]
                frames.append(subframe)
            previous = i
    result = pd.concat(frames)
    return result

# The following function takes a range of rows of a dataframe and returns a row with statistical information
# of that range of rows


def stat_contraction(frame, start, end):
    import numpy as np
    import math

    column_lists = []
    for j in frame.columns[2:]:
        column_lists.append(np.array(frame[j][start:end]))
    values = [frame.loc[start].at["time_id"]]
    for j in column_lists:
        values.append(np.max(j))
        values.append(np.min(j))
        values.append(np.sum(j) / len(j))
        values.append(math.sqrt(np.var(j)))
    values = np.array(values)
    values = values.reshape((1, len(values)))
    return values

# The following function takes a stock dataframe and returns the dataframe formed after replacing each time_id-groups
# with a row containing statistical information about it.


def contracted_frame(frame):
    import numpy as np
    import pandas as pd
    import math
    new_columns = ['time_id', 'bid_price1_max', 'bid_price1_min', 'bid_price1_av', 'bid_price1_sd', 'ask_price1_max',
                   'ask_price1_min', 'ask_price1_av', 'ask_price1_sd', 'bid_price2_max', 'bid_price2_min',
                   'bid_price2_av', 'bid_price2_sd', 'ask_price2_max', 'ask_price2_min', 'ask_price2_av',
                   'ask_price2_sd', 'bid_size1_max', 'bid_size1_min', 'bid_size1_av', 'bid_size1_sd', 'ask_size1_max',
                   'ask_size1_min', 'ask_size1_av', 'ask_size1_sd', 'bid_size2_max', 'bid_size2_min', 'bid_size2_av',
                   'bid_size2_sd', 'ask_size2_max', 'ask_size2_min', 'ask_size2_av', 'ask_size2_sd']
    contracted = pd.DataFrame(columns=new_columns)

    previous = 0
    n = frame.shape[0]
    for i in range(n - 1):
        if not frame.loc[i].at["time_id"] == frame.loc[i + 1].at["time_id"]:
            values = stat_contraction(frame, previous+1, i)
            temp = pd.DataFrame(values, columns=new_columns)
            contracted = pd.concat([contracted, temp])
            previous = i
        if i+1 == n-1:
            values = stat_contraction(frame, previous+1, i+1)
            temp = pd.DataFrame(values, columns=new_columns)
            contracted = pd.concat([contracted, temp])
            previous = i
    return contracted


