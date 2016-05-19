__author__ = 'fucus'

import logging
from config import Project
import sys
from shutil import copy2
import os


def restructure_img(result_csv_file_path, output_img_path):
    logging.info("restructure the image file by %s file to %s" % (result_csv_file_path, output_img_path))
    for j in range(10):
        output_img_type_path = "%s/c%d" % (output_img_path, j)
        if not os.path.exists(output_img_type_path):
            os.makedirs(output_img_type_path)

    count = 0
    for line in open(result_csv_file_path):
        if count % 1000 == 0:
            logging.info("process %d line of result csv file path now" % count)
        line = line.rstrip("\n")
        if count > 0:
            split_line = line.split(",")
            if len(split_line) != 11:
                logging.warning("can't extract info from line %s:%s" % (count, line))
            else:
                img_path = "%s/%s" % (Project.test_img_folder_path, split_line[0])

                for j in range(10):
                    if int(split_line[j+1]) > 0.5:
                        output_img_type_path = "%s/c%d/" % (output_img_path, j)
                        copy2(img_path, output_img_type_path)
                        break

        count += 1

if __name__ == '__main__':
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    level = logging.INFO
    if len(sys.argv) >= 2:
        level_name = sys.argv[1]
        level = LEVELS.get(level_name, logging.INFO)
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    csv_file_path = "%s/result/2016-05-15 08:25.csv" % Project.project_path
    output_img_path = "%s/../output_img/" % Project.project_path
    restructure_img(csv_file_path, output_img_path)

