from tkinter import *
from tkinter import ttk

from PIL import Image, ImageTk
import os
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from src.config import ASSETS_DIR
# Classes
class ToolBar:

    def __init__(self,parent,controller):
        # Toolbar Section
        self.controller = controller
        self.toolbar_fm = ttk.Frame(parent,relief='groove', borderwidth=10)
        self.toolbar_fm.grid(row = 0,column=0, sticky='nsew')
        self.toolbar_fm.rowconfigure(0,weight=1)
        # Loading Icons
        button_txt = ['execute','exit']
        self.icons ={}
        for txt in button_txt:
            self.icons[txt] = ImageTk.PhotoImage(Image.open(os.path.join(ASSETS_DIR, f'{txt}.png')))
        # Add Buttons to Toolbar Frame
        self.execute_bt = ttk.Button(self.toolbar_fm, image=self.icons['execute'], text= 'Execute', compound='top',command=self.controller.on_execute)
        #self.stop_bt = ttk.Button(self.toolbar_fm, image=self.icons['stop'], text= 'Stop', compound='top', command=self.controller.on_stop)
        self.exit_bt = ttk.Button(self.toolbar_fm, image=self.icons['exit'], text= 'Exit', compound='top',command=self.controller.on_exit)
        # Arrange Buttons 
        self.execute_bt.grid(row=0,column=0,padx=2,pady=2,sticky='nsew')
        #self.stop_bt.grid(row=0,column=1,padx=2,pady=2,sticky='nsew')
        self.exit_bt.grid(row=0,column=1,padx=5,pady=2,sticky='nsew')
        for i in range(2):
            self.toolbar_fm.columnconfigure(i,weight=0)
