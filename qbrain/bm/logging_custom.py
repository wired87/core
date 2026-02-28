import logging
import os


def cpr(*args):
    if os.name == "nt":
        print(*args)
    else:
        logging.info(" ".join(map(str, args)))  # Join args to a string.

