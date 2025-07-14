import time


def check_timestamp(timestamp):
    if isinstance(timestamp, (int, float)):
        if (time.time() - timestamp) > 0:
           #print("No AP possible")
            return False
    return True

