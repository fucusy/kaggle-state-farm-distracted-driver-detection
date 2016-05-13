__author__ = 'fucus'

from config import Project as p
import pandas as pd
import logging
import time
import os

def generate_result_file(name, y_result):
    df = pd.DataFrame({"img": name})

    # if y_result is a list of type
    y_result_matrix = [[]] * 10
    driver_type_list = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

    for i in range(len(y_result)):
        driver_type = y_result[i]
        if driver_type not in driver_type_list:
            logging.warning("the predicted driver type:%s of %s not in driver_type_list:%s, it's incorrect, force set"
                            " it to c0" % (driver_type, name[i], ",".join(driver_type_list)))
        else:
            driver_type = "c0"
        driver_index = driver_type_list.index(driver_type)

        for j in range(10):
            if j != driver_index:
                y_result_matrix[j].append(0)
            else:
                y_result_matrix[j].append(1)

    for i in range(len(driver_type_list)):
        df[driver_type_list[i]] = y_result_matrix[i]

    file_name = time.strftime("%Y-%m-%d %H:%M.csv")
    output_path = p.result_output_path.strip()
    final_path = ""
    if len(output_path) > 0 and output_path[0] == '/':
        final_path = output_path
    else:
        final_path = "%s/%s" % (p.project_path, output_path)
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    df.to_csv("%s/%s" % (final_path, file_name))