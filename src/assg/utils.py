from pathlib import Path

# determine root directory, this assumes that the the assg/utils.py
# file is in a child directory 2 levels below the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
