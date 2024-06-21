print('app start')
import sys
if getattr(sys, 'frozen', False):
    sys.stdout = open('output.txt', 'w', encoding='utf-8') # Compatiable with Console = False
    sys.stderr = open('error.txt', 'w', encoding='utf-8')   # Compatiable with Console = False

import src.mainapp
import multiprocessing
if __name__ =='__main__':
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support() # Enable Threading in C  
    src.mainapp.main()