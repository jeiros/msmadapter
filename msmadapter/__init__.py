# Taken from https://gist.github.com/st4lk/6287746
import logging
import logging.handlers

f = logging.Formatter(fmt='%(asctime)s %(levelname)s:%(name)s: %(message)s '
    '(%(filename)s:%(lineno)d)',
    datefmt="%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

handler.setFormatter(f)
handler.setLevel(logging.DEBUG)
root_logger.addHandler(handler)
