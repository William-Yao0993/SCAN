from tkinter import *
from tkinter import ttk
import os
from PIL import Image
from PIL.ImageTk import PhotoImage
from src.config import ASSETS_DIR, TEXTS_DIR,TEMP_DIR,IMG_SUFFIXS


class ContentWindow:
    def __init__(self,parent):
        # Fixed Component
        self.content_fm = ttk.Frame(parent, relief='solid', borderwidth=2)
        self.content_fm.grid(row=2,column=0,sticky='nsew',padx=10,pady=10)
        self.content_fm.columnconfigure(0,weight=1)
        self.content_fm.rowconfigure(0,weight=1)
        self.txts_suffix = ['upload.txt','run.txt','result.txt','export.txt'] 
        self.h_scroll = Scrollbar(self.content_fm,orient=HORIZONTAL)
        self.v_scroll = Scrollbar(self.content_fm,orient=VERTICAL)
        self.h_scroll.grid(row=1,column=0,columnspan=2,sticky='ew',padx=5,pady=5)
        self.v_scroll.grid(row=0,column=2, sticky='ns',padx=5,pady=5)
        
        self.canvas = Canvas(self.content_fm, yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        

        self.canvas.grid(row=0,column=0, sticky='nsew', columnspan=2)
        self.canvas.bind('<Configure>', self.resize_img) # Resize Current Display Image Only (Memory Saving) 
        self.canvas.bind_all('<MouseWheel>', self.on_mousewheel) # Mouse Scrollable
        self.txt_id = self.canvas.create_text(0,0, text=self.load_txt(os.path.join(TEXTS_DIR, self.txts_suffix[0])), fill= 'black', font=('Helvetica', 12), anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox(ALL)) # Centrelise the Text 
        self.v_scroll['command'] = self.canvas.yview
        self.h_scroll['command'] = self.canvas.xview
        
        # Addition Feature For different Section
        self.progress_bar = ttk.Progressbar(self.content_fm,orient='horizontal', length = 300, mode = 'indeterminate')
        prev_icon = PhotoImage(Image.open(os.path.join(ASSETS_DIR, 'prev.png')))
        next_icon = PhotoImage(Image.open(os.path.join(ASSETS_DIR, 'next.png')))
        self.icons= {
            'prev': prev_icon,
            'next': next_icon
        }
        self.prev_bt =ttk.Button(self.content_fm, image= self.icons['prev'],command=self.prev_img)
        self.next_bt = ttk.Button(self.content_fm, image=self.icons['next'],command=self.next_img)
        self.imgs_idx = None # Index of the current image shown
        self.tk_img = None # ImageTk PhotoGragh Displaying on the Screen
        self.pil_imgs = None
        self.img_id = None
        
        self.tree = ttk.Treeview(self.content_fm)
        
    def load_txt(self, path):
        with open(path, 'r') as file:
            return file.read()

    def update_frame(self,event):
        tab_index = event.widget.index('current') 
        self.canvas.itemconfig(self.txt_id, text= self.load_txt(os.path.join(TEXTS_DIR,self.txts_suffix[tab_index])))
        match(tab_index):
            case 1: # Run
                if self.img_id:
                    self.canvas.itemconfig(self.img_id, state = 'hidden')
                self.prev_bt.grid_forget()
                self.next_bt.grid_forget()
                if self.tree.grid_info():
                    self.tree.grid_forget()
                self.progress_bar.grid(row=3,column=0,sticky='e',padx=5, pady=5, columnspan=2)
            case 2: # Result 
                if self.img_id:
                    self.canvas.itemconfig(self.img_id, state = 'normal')
                #self.progress_bar.grid(row=3,column=0,sticky='e',padx=5, pady=5, columnspan=2)
                self.prev_bt.grid(row=2,column=0, sticky='wns', padx=5, pady=5)
                self.next_bt.grid(row=2, column=1, sticky='ens',padx=5,pady=5)
            case _:
                if self.img_id:
                    self.canvas.itemconfig(self.img_id, state = 'hidden')
                if self.tree.grid_info():
                    self.tree.grid_forget()
                self.canvas.grid(row=0,column=0, sticky='nsew', columnspan=2)
                self.progress_bar.grid_forget()
                self.prev_bt.grid_forget()
                self.next_bt.grid_forget()

    def show_img(self,index):
        #pass
        if 0 <= index:
            self.tree.grid_forget()
            self.canvas.grid(row=0,column=0, sticky='nsew', columnspan=2)
            self.v_scroll.config(command=self.canvas.yview)
            self.h_scroll.config(command=self.canvas.xview)
            self.imgs_idx = index
            current_img = self.pil_imgs[index]
            fitted_height = int(1.*self.canvas.winfo_height())
            fitted_width = int((fitted_height* current_img.width) / current_img.height)
            # fitted_width = int(0.9*self.canvas.winfo_width())
            # fitted_height = int((fitted_width*current_img.height)/current_img.width)
            current_img = current_img.resize((fitted_width,fitted_height), Image.Resampling.LANCZOS)
            self.tk_img = PhotoImage(current_img)
            if self.img_id is None:
                self.img_id = self.canvas.create_image(0,0,image=self.tk_img, anchor = 'nw')
            else:
                self.canvas.itemconfig(self.img_id, image = self.tk_img)
            self.canvas.config(scrollregion=self.canvas.bbox(ALL))
    def next_img(self):
        if self.imgs_idx < len(self.pil_imgs) -1:
            self.show_img(self.imgs_idx+1)
    def prev_img(self):
        if self.imgs_idx > 0:
            self.show_img(self.imgs_idx-1)
    def resize_img(self,event):
        if self.pil_imgs and self.imgs_idx is not None:
            current_img = self.pil_imgs[self.imgs_idx]
            # Resize 
            new_height = int(event.height)
            new_width = int((current_img.width * new_height) / current_img.height)
            current_img = current_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.show_img(self.imgs_idx)


    def show_table(self, df):
        self.canvas.grid_forget()
        self.tree.config(columns=list(df.columns), show='headings')
        self.tree.grid(row=0,column=0,sticky='nsew',columnspan=2)
        self.v_scroll.config(command=self.tree.yview)
        self.h_scroll.config(command=self.tree.xview)
        self.tree.config(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        # Add column headings
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=50)

        # Add data to the treeview
        from pandas import isna
        for index, row in df.iterrows():
            line = ['' if isna(i) else i for i in row]
            self.tree.insert("",END, values=line)

        self.canvas.config(scrollregion=self.canvas.bbox('all'))
    # Event Binding Function
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), UNITS)
    def start_progress_bar(self):
        self.progress_bar.start()
    def stop_progress_bar(self):
        self.progress_bar.stop()
    def init_imgs(self,imgs):
        self.pil_imgs=imgs
    def load_imgs(self,dir_path):
        imgs=[]   
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(IMG_SUFFIXS):
                file_path = os.path.join(dir_path,filename)
                with Image.open(file_path) as img:
                    imgs.append(img.copy())
        return imgs
    def img_directory_on_change(self,event):
        curr_dir =event.widget.get()
        full_path_curr_dir = os.path.join(TEMP_DIR,curr_dir)
        imgs = self.load_imgs(full_path_curr_dir)
        self.init_imgs(imgs)
        self.show_img(0)

   