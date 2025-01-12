import json
import time

import numpy as np

from elfantasy.config import Configuration


# timer decorator
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


# generate timetag YYYYMMDDHHmmSS
def get_timetag():
    return time.strftime("%Y%m%d%H%M%S")


# softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


# funtion to read json file
config = Configuration()
data_dir_datalog = config.data_dir_datalog


def read_datalog():
    with open(data_dir_datalog) as f:
        return json.load(f)


def write_datalog(data):
    with open(data_dir_datalog, "w") as f:
        json.dump(data, f, indent=4)


def update_datalog(data):
    datalog = read_datalog()
    datalog.update(data)
    write_datalog(datalog)
