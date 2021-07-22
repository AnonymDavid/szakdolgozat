import zipfile
from sys import argv


if len(argv) < 3:
    exit("Not enough arguments: python compress.py [FOLDER_NAME] [OUTPUT_NAME]")

ZIP_NAME = argv[2]
FILE_FOLDER = argv[1]
FILE_NAMES = ["[Content_Types].xml", "_rels/.rels", "circuitdiagram/Document.xml", "docProps/core.xml"]

compression = zipfile.ZIP_DEFLATED

zf = zipfile.ZipFile(ZIP_NAME, mode="w")

try:
    for file_to_write in FILE_NAMES:
        zf.write(FILE_FOLDER+"\\"+file_to_write, file_to_write, compress_type=compression)
except FileNotFoundError as e:
    print(f' *** Exception occurred during zip process - {e}')
finally:
    zf.close()