import os
import pickle
import logging

logger = logging.getLogger()

# Directory in which to store precomputed accountant tables.
accountant_tables_dir = os.path.expanduser('~/.pytorch_privacy/cache')
if not os.path.isdir(accountant_tables_dir):
    os.makedirs(accountant_tables_dir)


def set_accountant_tables_dir(new_dir):
    global accountant_tables_dir
    accountant_tables_dir = new_dir


def grab_pickled_accountant_results(filename):
    try:
        filepath = os.path.join(accountant_tables_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as file:
                return True, pickle.load(file), filepath
        else:
            return False, None, filepath
    except (IOError, FileNotFoundError):
        logger.error(f"Issue with opening Pickled Accountant: {filename}")
