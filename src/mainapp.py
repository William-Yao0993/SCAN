from ttkthemes import ThemedTk
from src.toolbar import ToolBar
from src.navi import NaviBar
from src.content import ContentWindow
from tkinter import messagebox
from src.utils.predict import predict_with_threads,draw_distribution_plots,export_excel,export_imgs,export_plots
from concurrent.futures import ThreadPoolExecutor
import sys
from src.config import TEMP_DIR, CACHE_DIR, STOMATA_DENSITY,STOMATA_SIZE,PORE_SIZE,FOLDER_ID
import time
import os
import shutil
# Main class encapsulates all GUI design

class MainApp:
    def __init__(self,root):
        self.root = root
        self.controller = self.AppController(self)
        self.toolBar = ToolBar(root,self.controller)
        self.naviBar = NaviBar(root)
        self.contentView = ContentWindow(root)
        self.naviBar.navi_nb.bind('<<NotebookTabChanged>>', lambda event: self.contentView.update_frame(event))
        self.naviBar.result_tb.dir_dd.bind('<<ComboboxSelected>>',lambda event: self.contentView.img_directory_on_change(event))
        root.protocol('WM_DELETE_WINDOW', self.controller.on_exit)

    # App Controller 
    class AppController:
        def __init__(self,app):
            self.mainapp = app
            self.data_path = None
            self.future = None # Thread management for BackEnd 
            self.statistics = None # pd.Dataframe 
            self.plots = None # dict  
            self.imgs = None # list
            self.sb_unit = None
        def on_execute(self):

            navi = self.mainapp.naviBar
            curr_tb = navi.navi_nb.index(navi.navi_nb.select())
            
            
            match(curr_tb):
                case 0: # Upload

                    self.data_path = navi.upload_tb.execute()
                case 1: # Run 
                    try:
                        if not self.data_path:
                            messagebox.showerror(title='Run Section Error', message='Fail to Load Model and Data, Please Go Back Upload Section')
                        
                        run_pore,conf,sb_lenth, sb_unit,sb_pxl = navi.run_tb.execute()
                        self.sb_unit = sb_unit
                        self.start_threading(predict_with_threads, self.data_path,run_pore,conf,sb_lenth, self.sb_unit,sb_pxl)
                        self.future_callback(self.set_statistics)
                    except TypeError as te:
                        messagebox.showerror(title='Run Tab Error', message=te)
                    except Exception as e: 
                        messagebox.showerror(title='Run Tab Error', message='Something unexpected happens, please report on github or author')
                case 2: # Result
                    if (self.statistics is None) and (len(self.statistics) !=0):
                        messagebox.showerror(title='Visualisation Failure',message='Fail to load the statistics, please rerun the model')
                    else:
                        self.plots = draw_distribution_plots(self.statistics,FOLDER_ID,[STOMATA_DENSITY+f'({self.sb_unit}²)',STOMATA_SIZE+f'({self.sb_unit}²)',PORE_SIZE+f'({self.sb_unit}²)'],self.sb_unit)
                        temps = [subdir for subdir in os.listdir(TEMP_DIR) if os.path.isdir(os.path.join(TEMP_DIR,subdir))]
                        self.mainapp.naviBar.result_tb.set_dropdown(temps)
                        
                        # Visualise 
                        self.visualise()
                        navi.result_tb.control_var.trace_add('write',self.visualise)    
                case 3: # Export

                    if self.plots is None or self.statistics is None:
                        messagebox.showerror(title='Export Error', message='Execute Result Before Export.')
                    result = navi.export_tb.execute()
                    if result is not None:
                        img_var, plt_var, exl_var = result
                        export_dir = os.path.join(self.data_path,'predict')
                        if os.path.exists(export_dir):
                            shutil.rmtree(export_dir)
                        os.mkdir(export_dir)
                        if img_var:
                            export_imgs(export_dir)
                        if plt_var:
                            pass
                            export_plots(self.plots, export_dir)
                        if exl_var:
                            export_excel(self.statistics,export_dir)
                        messagebox.showinfo(title='Export Windows', message=f'Export to {export_dir}')


        # def on_stop(self):
        #     #print('Stop Clicked!')
        #     if self.future:
        #         self.flush()
        #         self.mainapp.contentView.stop_progress_bar()
        #         messagebox.showinfo(title='Stop Windows', message='All the Running Tasks Have Stopped')
        #     if os.listdir(TEMP_DIR):
        #         for dir in os.listdir(TEMP_DIR):
        #             if os.path.isdir(dir):
        #                 shutil.rmtree(dir)
        def on_exit(self):
            #print('Exit Clicked')
            if os.listdir(CACHE_DIR):
                for temp in os.listdir(CACHE_DIR):
                    shutil.rmtree(os.path.join(CACHE_DIR,temp))
            self.flush()
            self.mainapp.root.destroy()
            os._exit(0) # Force Exit Program Including All Threads

        def flush(self):
                self.future = None
                self.data_path = None
                self.results = None
                self.future = None 
                self.df = None 
                self.img_idxs = None 
                self.plots = [] 
        # Threading Functions  
        def start_threading(self,callback, *args, **kwargs):

            executor = ThreadPoolExecutor()
            self.future = executor.submit(callback, *args, **kwargs)

        def future_callback(self,callback):
            #start_time = time.time()
            try:
                if self.future is not None:
                    content = self.mainapp.contentView
                    content.start_progress_bar()
                    if self.future.done():
                        result = self.future.result()
                        callback(result) # Save Result in Controller
                        #end_time = time.time()
                        #print(f'Future takes {end_time-start_time} seconds')
                        self.future = None # Clear Future
                        content.stop_progress_bar()
                        messagebox.showinfo(title='Progress Result', message='Execution Completed, Please Start the Next Operation')
                    else:
                        content.content_fm.after(500, self.future_callback, callback)
            except TypeError as te:
                content.stop_progress_bar()
                messagebox.showerror(title='Input type error', message=str(te))
            except Exception:
                content.stop_progress_bar()
                messagebox.showerror(title='Run Tab Error', message='Something unexpected happens, please report on GitHub or author')
            
        # # Callbacks to Present Storage from Threads 
        def set_statistics(self,statistics):
            self.statistics = statistics
            
        def visualise(self, *args):
            option = self.mainapp.naviBar.result_tb.control_var.get()
            match option:
                case 0: # Images
                    curr_dir = self.mainapp.naviBar.result_tb.dir_dd.get()
                    full_path_curr_dir = os.path.join(TEMP_DIR,curr_dir)
                    imgs = self.mainapp.contentView.load_imgs(full_path_curr_dir)
                    self.mainapp.contentView.init_imgs(imgs)
                    self.mainapp.contentView.show_img(0)
                case 1: # Plots
                    self.mainapp.contentView.init_imgs(list(self.plots.values()))
                    self.mainapp.contentView.show_img(0) 
                case 2: # Table
                    self.mainapp.contentView.show_table(self.statistics)

# GUI Design
def main():
    #-------------------------------------------------------------------------------
    # Main Windows
    if getattr(sys, 'frozen', False):
        import pyi_splash
        pyi_splash.update_text('Starting...')
    start_time = time.time()
    #print(f'GUI Starting: {start_time}')
    root = ThemedTk(theme='plastik')
    scr_width = int(root.winfo_screenwidth() *0.65)
    scr_height = int(root.winfo_screenheight() * 0.9)

    root.geometry(f"{scr_width}x{scr_height}")
    root.attributes()
    root.title('SCAN')
    root.columnconfigure(0,weight=1)
    root.rowconfigure(0,weight=0) # Button Section 
    root.rowconfigure(1,weight=0) # Navigation Bar + Tab
    root.rowconfigure(2,weight=10) # Main Content

    gui = MainApp(root)
    #print(Tcl().eval('set tcl_platform(threaded)'))
    if getattr(sys, 'frozen', False):
        import pyi_splash
        pyi_splash.close()
    print(f'GUI Start In Total: {time.time() - start_time}')

    root.after(1, bring_to_front, root)
    root.mainloop()
    

def bring_to_front(window):
    window.lift()
    window.focus_force()