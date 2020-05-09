import re

import PyPDF2
# 破解pdf的密码，pdf密码是一个大写的英文单词
with open('./code/Python_Tricks_encrypted.pdf', 'rb') as pdf_file_stream:
    reader = PyPDF2.PdfFileReader(pdf_file_stream)
    with open('./code/dictionary.txt', 'r') as txt_file_stream:
        file_iter = iter(lambda: txt_file_stream.readline(), '')
        for word in file_iter:
            word = re.sub(r'\s', '', word)
            if reader.decrypt(word):
                print(word)
                break

