#!/usr/bin/env python
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

from docqa.commands import main  # pylint: disable=wrong-import-position
# Custom obqa modules for registering models, etc
import docqa.models
import docqa.data.dataset_readers

if __name__ == "__main__":
    main(prog="python -m docqa.run")
