import os

def read_txt(name):
    try:
        f = open(name, 'r', encoding="utf8")
        paragraphs = f.readlines()        # newline을 하나의 문단 구분으로 봄
    except:
        f = open(name, 'r', encoding="cp949")
        paragraphs = f.readlines()        # newline을 하나의 문단 구분으로 봄
    f.close()
    return paragraphs

