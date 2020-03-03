#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

def format_files(path):
    for root, dirs, files in os.walk(path):
        fname = []
        for file in files:  
            if root.find("sysroot")>=0:
                continue
            if root.find("install")>=0:
                continue
            if root.find ("CMakeFiles")>=0:
                continue
            if os.path.splitext(file)[1] == '.cpp' or os.path.splitext(file)[1] == '.c' or \
               os.path.splitext(file)[1] == '.hpp' or os.path.splitext(file)[1] == '.h' :
                fname = os.path.join(root, file)
                if fname.find("include/any.hpp")>=0:
                    continue

                print("dos2unix %s" %(fname))
                os.system("dos2unix %s" %(fname))

                print("clang-format -style=file -i %s" %(fname))
                os.system("clang-format -style=file -i %s" %(fname))

if __name__ == '__main__':
    path = './'
    format_files(path)
