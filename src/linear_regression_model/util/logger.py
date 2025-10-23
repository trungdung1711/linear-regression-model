import logging

logger = logging.getLogger("Model")

handler = logging.StreamHandler()

logger.addHandler(handler)

logger.setLevel(logging.WARNING)
