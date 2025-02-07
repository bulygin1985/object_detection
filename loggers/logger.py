import logging
import os
import sys
import time

LOGGER_NAME = "object_detection"
LOG_DIR = "../logs"

script_name, _ = os.path.splitext(os.path.basename(sys.argv[0]))


logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

log_file = os.path.join(
    LOG_DIR,
    f"{script_name}_{time.strftime('%Y-%m-%d_%H:%M:%S.log', time.localtime(time.time()))}",
)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.WARNING)
sh.setFormatter(formatter)
logger.addHandler(sh)
