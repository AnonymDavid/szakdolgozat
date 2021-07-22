import zipfile

compression = zipfile.ZIP_DEFLATED

ZIP_NAME = "test.cddx"
FILE_NAMES = ["[Content_Types].xml", "_rels/.rels", "circuitdiagram/Document.xml", "docProps/core.xml"]

zf = zipfile.ZipFile(ZIP_NAME, mode="w")

try:
    for file_to_write in FILE_NAMES:
        zf.write(file_to_write, file_to_write, compress_type=compression)
except FileNotFoundError as e:
    print(f' *** Exception occurred during zip process - {e}')
finally:
    zf.close()