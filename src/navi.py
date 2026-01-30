from tkinter import ttk,filedialog,messagebox,END,BooleanVar,StringVar,IntVar,DoubleVar
import os
from src.config import SB_UNIT,SB_IN_PIXEL,SB_LENGTH,MAX_THREADS

class NaviBar:
    def __init__(self, parent):
        self.navi_nb = ttk.Notebook(parent)
        self.navi_nb.grid(row=1, column=0,sticky='ew', padx=10, pady=5)
        self.upload_tb = self.UploadTab(self.navi_nb)
        self.run_tb = self.RunTab(self.navi_nb)
        self.result_tb = self.ResultTab(self.navi_nb)
        self.export_tb = self.ExportTab(self.navi_nb)

    class UploadTab:
        def __init__(self, notebook):
            self.upload_fm = ttk.Frame(notebook,padding='5 5 12 12') 
            notebook.add(self.upload_fm, text='upload')
            self.data_lb = ttk.Label(self.upload_fm, text="Import Data:")
            self.data_lb.grid(row=0,column=2,sticky='ew', padx=5,pady=5)
            # Data Import Entry & Button
            self.data_et =ttk.Entry(self.upload_fm)
            self.data_et.grid(row=0,column=3,sticky='ew',padx=5,pady=2)
            self.data_bt = ttk.Button(self.upload_fm, text='Browse',command=self.choose_dir)
            self.data_bt.grid(row=0,column=4,padx=5,pady=2)
            self.upload_fm.rowconfigure(0,weight=1)
            self.upload_fm.columnconfigure(3,weight=1)
        def choose_dir(self):
            '''
                Function to choose directory and update the status text
            '''
            dir = filedialog.askdirectory()
            if dir:
                # Update the entry box with the selected directory path
                self.data_et.delete(0,END)
                self.data_et.insert(0,dir)

        def execute(self):
            """
            Validate the model path and data path.
            Return: 
                (Model Path, Directory Path)
            """
            # print('Before:', controller.data)
            data_path = self.data_et.get().strip()

            # Check Model Dropdown Status 
            if not os.path.exists(data_path):
                messagebox.showerror(title='Setup Error', message='Plase Select Data to Start.')
                return None
            else:
                messagebox.showinfo(title='Setup Completed', message='Data Successfully Imported! Please Go to Run Section.')
            
            return data_path 
    class RunTab:
        def __init__(self, notebook):
            self.run_fm = ttk.Frame(notebook,padding='5 5 12 12')
            notebook.add(self.run_fm,text='run')

            # Toggle Widgets 
            self.pore_lb = ttk.Label(self.run_fm,text='Aperture Size')
            self.pore_lb.grid(row=0,column=0, sticky='ew',padx=5,pady=5)
            self.pore_val =BooleanVar(value= True)
            self.pore_cb = ttk.Checkbutton(self.run_fm, variable=self.pore_val)
            self.pore_cb.grid(row=0,column=1,sticky='ew',padx=2,pady=2)

            self.conf_lb = ttk.Label(self.run_fm, text='confidence')
            self.conf_lb.grid(row=0,column=2, sticky='e')
            self.conf_val =DoubleVar(value='0.25')
            self.conf_et = ttk.Entry(self.run_fm, width=10, textvariable=self.conf_val)
            self.conf_et.grid(row=0,column=3,sticky='e')

            self.sbl_lb = ttk.Label(self.run_fm, text='Scale bar length: ')
            self.sbl_lb.grid(row=1,column=0, sticky='ew')
            self.sbl_val = DoubleVar(value=SB_LENGTH)
            self.sbl_et = ttk.Entry(self.run_fm, width=5, textvariable=self.sbl_val)
            self.sbl_et.grid(row=1,column=1,sticky='ew')
            self.unit_val = StringVar(value=SB_UNIT)
            self.unit_et = ttk.Entry(self.run_fm, width=5, textvariable=self.unit_val)
            self.unit_et.grid(row=1,column=2,sticky='ew')

            self.pxl_lb = ttk.Label(self.run_fm, text='In Pixel: ')
            self.pxl_lb.grid(row=1,column=3, sticky='ew')
            self.pxl_val = DoubleVar(value=SB_IN_PIXEL)
            self.pxl_et = ttk.Entry(self.run_fm, width=5, textvariable=self.pxl_val)
            self.pxl_et.grid(row=1,column=4,sticky='ew')

            # Additional Double input
            self.threads_lb = ttk.Label(self.run_fm, text='Threads: ')
            self.threads_lb.grid(row=1,column=6, sticky='ew')
            self.threads_val = IntVar(value=MAX_THREADS)
            self.threads_et = ttk.Entry(self.run_fm, width=5, textvariable=self.threads_val)
            self.threads_et.grid(row=1,column=7,sticky='ew')

            
        def execute(self):
            '''
                Check Confidence and Stream Mode Input back to Controller 
                Return:
                    (Stream, Confidence, Scale bar length, Scale bar unit, Scale bar length in pixel, Threads) 
            '''
            conf = self.conf_val.get()
            if conf >= 1 or conf <=0.001:
                messagebox.showerror(title='Run Section Error',message='Confidence Should in Range 0.001-1 Exclusively')
                return None

            # Validate threads input
            try:
                threads = int(self.threads_val.get())
                if threads < 1:
                    messagebox.showerror(title='Run Section Error', message='Threads must be >= 1')
                    return None
            except Exception:
                messagebox.showerror(title='Run Section Error', message='Threads must be an integer')
                return None

            return self.pore_val.get(), conf, self.sbl_val.get(),self.unit_val.get(),self.pxl_val.get(), threads

    class ResultTab:
        def __init__(self,notebook):
            # Results Section UI 
            self.result_fm = ttk.Frame(notebook,padding='5 5 12 12')
            notebook.add(self.result_fm,text='result')
            self.result_fm.columnconfigure(4, weight=1) # Span Directory DropDown 
            self.show_lb = ttk.Label(self.result_fm, text='Show:')
            self.show_lb.grid(row=0,column=0,sticky='e')

            self.control_var = IntVar()
            
            self.img_rd = ttk.Radiobutton(self.result_fm, text= 'Images',variable=self.control_var, value=0)
            self.plot_rd = ttk.Radiobutton(self.result_fm,text='Plots',variable=self.control_var,value=1)
            self.table_rd = ttk.Radiobutton(self.result_fm,text='Table',variable=self.control_var,value=2)

            self.img_rd.grid(row=0,column=1,padx=10,pady=10,sticky='w')
            self.plot_rd.grid(row=0,column=2,padx=10,pady=10,sticky='w')
            self.table_rd.grid(row=0,column=3,padx=10,pady=10,sticky='w')
            self.dirs = None
            self.dir_dd = ttk.Combobox(self.result_fm,width=150)
            self.control_var.trace_add('write', self.toggle_dropdown)
        def execute(self):
            '''
                Visualise and Analysise from Results,
                Generating Images, Plots or Excel Based on User Requests 
                
                Return: 
                RadioButton Value to Controller to Visualise 
            '''
            return self.control_var.get()
        
        def toggle_dropdown(self, *args):
            '''
                Toggle On Directory dropdown for Image Option,
                Toggle Off for the Rest
            '''
            # print('Switch option!')
            if self.control_var.get() != 0 or self.dirs == None:
                self.dir_dd.grid_forget()
            else:
                self.dir_dd.grid(row=0,column=4, padx=10, pady=10,sticky='e')
        def set_dropdown(self,dirs):
            print(dirs)
            self.dirs = dirs
            self.dir_dd.configure(values=self.dirs)
            self.dir_dd.set(dirs[0])
            self.toggle_dropdown()
        
 
    class ExportTab:
        def __init__(self,notebook):
            # Export Section UI 
            self.export_fm = ttk.Frame(notebook, padding='5 5 12 12')
            notebook.add(self.export_fm, text = 'export')
            
            # Image checkbox
            self.img_lb = ttk.Label(self.export_fm, text='Images')
            self.img_lb.grid(row=0,column=0, sticky='nsew')
            self.img_var = BooleanVar(value= False)
            self.img_cb = ttk.Checkbutton(self.export_fm, variable=self.img_var)
            self.img_cb.grid(row=0, column=1,sticky='nsew')

            # Plot Checkbox
            self.plt_lb = ttk.Label(self.export_fm, text='Plots')
            self.plt_lb.grid(row=0,column=2, sticky='nsew')
            self.plt_var = BooleanVar(value= False)
            self.plt_cb = ttk.Checkbutton(self.export_fm, variable=self.plt_var)
            self.plt_cb.grid(row=0, column=3,sticky='nsew')

            # Excel Checkbox
            self.exl_lb = ttk.Label(self.export_fm, text='Excel')
            self.exl_lb.grid(row=0,column=4, sticky='nsew')
            self.exl_var = BooleanVar(value= False)
            self.exl_cb = ttk.Checkbutton(self.export_fm, variable=self.exl_var)
            self.exl_cb.grid(row=0, column=5,sticky='nsew')                        
        def execute(self):
            '''
                Send the Information Back to Controller Based on User Selected 

                Return: (Tuple) 
                Image(Boolean), Plot(Boolean), Excel(Boolean)
            '''
            img_var = self.img_var.get()
            plot_var = self.plt_var.get()
            exl_var = self.exl_var.get()

            if not (img_var or plot_var or exl_var): 
                messagebox.showerror(title='Export Error', message='None of The Options Selected, Please Select At Least One Option') 
                return None
            return img_var, plot_var, exl_var   