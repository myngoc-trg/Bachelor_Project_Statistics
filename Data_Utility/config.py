import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

TRAIN_DIR = os.path.join(PROJECT_ROOT, "Data/Size-data/Sorted_224_sizeTrain")

TEST_DIR = os.path.join(PROJECT_ROOT, "Data/Size-data/Sorted_224_sizeTest")


EXCEL_PATH = os.path.join(PROJECT_ROOT, "Data/Size-data/Size_features/size_data.xlsx")