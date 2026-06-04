import zipfile
import xml.etree.ElementTree as ET

z = zipfile.ZipFile('ReqCluster_PRD.docx')
tree = ET.parse(z.open('word/document.xml'))
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
paragraphs = tree.findall('.//w:p', ns)
for p in paragraphs:
    text = ''.join(node.text or '' for node in p.findall('.//w:t', ns))
    if text.strip():
        print(text)
