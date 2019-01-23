"""
@Author: SBAITI Abdelkhalek <abdelkhalek.sbaiti@gmail.com>
"""

import app
import sys


def prediction(message):
    result = app.predict(message)
    return result

if __name__ == '__main__':
    if(len(sys.argv)<2):
        print("Usag : prediction \" message txt \"")
    else:
        txt = sys.argv[1]
        print(txt)
        r = prediction(txt)
        print("this message is ",r," ." )

