import xml.etree.ElementTree as ET

def getChess():
    tree = ET.parse("../dataset/GrayCodes_HighRes/graycodes_chess.xml")
    files = list(tree.find("images").itertext())[0].split("\n")
    return files[1:-1]

def getView0():
    tree = ET.parse("../dataset/GrayCodes_HighRes/graycodes_view0.xml")
    files = list(tree.find("images").itertext())[0].split("\n")
    return files[1:-1]

def getView1():
    tree = ET.parse("../dataset/GrayCodes_HighRes/graycodes_view1.xml")
    files = list(tree.find("images").itertext())[0].split("\n")
    return files[1:-1]