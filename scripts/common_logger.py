import sys
import logging


# Init logger for both file and stdout
def init_logger(file_name: str):
    log_format = "%(asctime)s [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(filename=file_name, level=logging.DEBUG, format=log_format)
    formatter = logging.Formatter(log_format)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.getLogger().addHandler(stream_handler)

