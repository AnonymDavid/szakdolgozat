from zipfile import ZipFile

with ZipFile('output.zip', mode='w') as zf:
    zf.write("output/_rels/.rels", "_rels/.rels")
    zf.write("output/circuitdiagram/Document.xml", "circuitdiagram/Document.xml")
    zf.write("output/docProps/core.xml", "docProps/core.xml")
    zf.write("output/[Content_Types].xml", "[Content_Types].xml")
