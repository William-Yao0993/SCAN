from ultralytics import YOLO
import pandas as pd
import numpy as np 
import os 
from .area_calculation import line_detect,masks_calculator,img_area
from pathlib import Path
import warnings
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..config import *
import re
##-----------------------------------------------------------------------------------
# Verison 1.1
# Change Prection mode to Stream (Results type: Generator), less memory comsumption
# Analysis in each result immediately
# Intergrate analyzed data in one frame
#------------------------------------------------------------------------------------
def predict_with_threads(dir_path,run_pore,conf,sb_lenth,sb_unit,sb_pxl):

    '''
            Start Multiple Model Instances to Conduct Inference in Threads,
        Wait all Threads and Gather the Results 

        Determine Number of Threads:
            Min (CPU Cores *4, Number of Sub-Folders)
        Return:
        Statistic results in all tasks(Threads) as Pandas.DataFrame 

    '''
    start_time=time.time()
    tasks_dict = directory_walk(dir_path)
    tasks = list(tasks_dict.keys())
    total_imgs = sum(list(tasks_dict.values())) # For updating progress bar
    if total_imgs == 0:
        raise TypeError(f'There is no files found with following suffixes {IMG_SUFFIXS}')
    threads_count = min(os.cpu_count()*4,len(tasks))
    grouped_tasks= bin_split(tasks,threads_count)
    print (f'Run {threads_count} threads and for groups: {grouped_tasks}')
    with ThreadPoolExecutor(threads_count) as executor:
        futures= [executor.submit(predict_and_analyse,group,run_pore,conf,sb_lenth, sb_unit,sb_pxl) for group in grouped_tasks]
        statistics =[]
        for future in as_completed(futures):
            try:
                result = future.result()
                #print(result)
                statistics.extend(result)
            except Exception as e:
                warnings.warn(e)
                return None
    print(f'{threads_count} threads prediction time : {time.time() - start_time}', flush=True)
    return pd.DataFrame(statistics)
def predict_and_analyse(grouped_tasks,run_pore,conf,sb_lenth, sb_unit,sb_pxl):
    '''
        Inference each task in the group with model and generate post-processing data in sequential manner
        Return: A list of dict of current group in a flatten way 
        e.g. ['stomata info * N','img Summary * N', 'dir summary 1',
            'stomata info * N','img Summary * N', 'dir summary 2',...] 
    '''
    data =[]
    # Run each task in the group  
    for dir_path in grouped_tasks:
        dir_name = Path(dir_path).name
        model_instance = YOLO(STOMATA_MODEL)
        results = model_instance.predict(
            dir_path,
            imgsz = 640,
            save=True,
            stream=True, 
            conf = conf, 
            verbose= False,
            name= Path(dir_path).name,
            project= TEMP_DIR,
            )
        
        # Process stomata results 
        for i,r in enumerate(results):
            img = r.orig_img
            img_name = Path(r.path).stem
            bboxes = r.boxes
            masks = r.masks
            
            # Inference Exist  
            if masks and bboxes:
                
                default_ratio =sb_lenth/sb_pxl
                ratio_to_unit = line_detect(img,sb_lenth,default_ratio)
                #print(ratio_to_unit)
                area_in_mm2 = img_area(img, ratio_to_unit)
                #print(area_in_mm2)
                masked_areas= masks_calculator(masks,ratio_to_unit)
                #print(len(masked_areas))
                pore_areas=np.full_like(masked_areas,fill_value=np.nan) # iterable placeholder
                
                # Run Pore Segment model 
                if run_pore:
                    cropped_imgs= crop(img,bboxes)
                    # temp_path = os.path.join(TEMP_DIR,f'crops_{img_name}')
                    # if not os.path.exists(temp_path):
                    #     os.mkdir(temp_path)
                    # for i,arr in enumerate(cropped_imgs):
                    #     bgr = arr[...,::-1]
                    #     img = Image.fromarray(bgr,'RGB')
                    #     img.save(os.path.join(temp_path,f'{i}.jpg'))
                    pore_model = YOLO(PORE_MODEL)
                    pore_results = pore_model.predict(
                    cropped_imgs,
                    verbose = False,
                    #imgsz = 108,
                    save=True,
                    max_det=1, # Each Stomata has one pore
                    stream=True, 
                    conf = conf, 
                    name= f'crops_{img_name}',
                    project= os.path.join(TEMP_DIR,Path(dir_path).name)
                    )
                    pore_areas = np.array([masks_calculator(pr.masks,ratio_to_unit) for pr in pore_results]).flatten()

                # Summary current Image
                stomata_list = process_stomata_data(dir_name,img_name,masked_areas,pore_areas,sb_unit)
                data.extend(stomata_list)
                img_summary = process_image_data(stomata_list,sb_unit,area_in_mm2)
                data.append(img_summary)
        # Summary current task (directory) 
        dir_summary = process_directory_data(data,sb_unit)
        if dir_summary is not None:
            data.append(dir_summary) 
    return data
# Statistic results structure functions  
def process_directory_data(data: list,sb_unit) -> dict:
    if data:
        df = pd.DataFrame(data)
        #df = df.dropna(axis=0,subset=[STOMATA_ID])
        dir_name = df[FOLDER_ID].iloc[-1]
        summary = {
            FOLDER_ID: f'Folder Summary: {dir_name}',
            STOMATA_DENSITY+'_Mean': df[STOMATA_DENSITY].mean(),
            STOMATA_DENSITY+'_SE': df[STOMATA_DENSITY].sem(),
            STOMATA_SIZE+f'({sb_unit}²)'+'_Mean': df[STOMATA_SIZE+f'({sb_unit}²)'].mean(),
            STOMATA_SIZE+f'({sb_unit}²)'+'_SE': df[STOMATA_SIZE+f'({sb_unit}²)'].sem(),
            PORE_SIZE+f'({sb_unit}²)'+'_Mean': df[PORE_SIZE+f'({sb_unit}²)'].mean(),
            PORE_SIZE+f'({sb_unit}²)'+'_SE': df[PORE_SIZE+f'({sb_unit}²)'].sem(),
            }
        return summary
    return None
def process_image_data(data:list,sb_unit,area_in_mm2) -> dict:
    '''
        Process all Stomata Information, generate Image summary and return as a list
        Return:
        dict:Image Summary
    '''
    df = pd.DataFrame(data)
    dir_name, img_name = df[FOLDER_ID].iloc[-1], df[IMAGE_ID].iloc[-1]
    image_summary = {
        FOLDER_ID: dir_name,
        IMAGE_ID: f'Image Summary: {img_name}',
        STOMATA_COUNT: df[STOMATA_ID].count(),
        STOMATA_DENSITY: df[STOMATA_ID].count() / area_in_mm2,
        STOMATA_SIZE+f'({sb_unit}²)'+'_Mean': df[STOMATA_SIZE+f'({sb_unit}²)'].mean(),
        STOMATA_SIZE+f'({sb_unit}²)'+'_SE': df[STOMATA_SIZE+f'({sb_unit}²)'].sem(),
        PORE_SIZE+f'({sb_unit}²)'+'_Mean': df[PORE_SIZE+f'({sb_unit}²)'].mean(),
        PORE_SIZE+f'({sb_unit}²)'+'_SE': df[PORE_SIZE+f'({sb_unit}²)'].sem()
    }
    return image_summary
def process_stomata_data(dirID,imgID,masked_areas,pore_areas,sb_unit) -> list:    
    '''
        Process and create statistic result for each stomata and return as a list
        Return:
        [stomata_info1,stomata_info2,...]
    '''
    stomata_list=[]
    for i,(masked_area,pore_area) in enumerate(zip(masked_areas,pore_areas)):
        stomata_list.append({
                        FOLDER_ID: dirID,
                        IMAGE_ID: imgID,
                        STOMATA_ID: int(i+1),
                        STOMATA_SIZE+f'({sb_unit}²)': masked_area,
                        PORE_SIZE+f'({sb_unit}²)': pore_area
                    })
    return stomata_list

# Drawing functions
def draw_distribution_plots(data: list| pd.DataFrame ,x:str,ys:list,sb_unit):
    '''
        Draw Mutiple BoxPlot for x to show its distribution in each Y values in ys list and return a list of PIL Images

        Return: A list of PIL images
        e.g.: {plt name 1: PIL_1, plt name 2: PIL_2,...}
    '''
    import matplotlib
    matplotlib.use('Agg') # Non-GUI Agg Backend

    plots = {y:draw_distribution_plot(data,x,y) for y in ys}
    plots['regression'] = draw_regression_plot(data,sb_unit)
    return plots
def draw_distribution_plot(data: list| pd.DataFrame ,x:str,y:str):
    '''
        Draw BoxPlot for x to show its distribution in Y values and return a PIL Image

        Return:
        Plot Image (format: PIL)
    '''
    if isinstance(data,list):
        df = pd.DataFrame(data)
    else:
        df = data
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig, ax = plt.subplots(figsize = (16,10))
    df_cp = df.dropna(axis=0,subset=[x,y],inplace=False).copy()
    sorted_categories=None

    if df_cp[x].apply(lambda s: re.search(r'\d+', s)).notnull().any():                           
        df_cp['x_numeric'] = df_cp[x].apply(lambda s: int(re.search(r'\d+', s).group())) 
        sorted_categories = df_cp.sort_values('x_numeric')[x].unique()
    sns.boxenplot(x=x, y=y, data=df_cp, ax=ax,order=sorted_categories) # More qualtiles 
    
    ax.set_title(f'{y} Distribution',fontsize = BIG_FONT_SIZE)
    ax.set_ylabel(y,fontsize=MEDIUM_FONT_SIZE)
    plt.xticks(rotation=45,fontsize=MEDIUM_FONT_SIZE)
    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    plot = Image.frombytes('RGB', canvas.get_width_height(),canvas.tostring_rgb())
    plt.close(fig)
    return plot 

def draw_regression_plot(data: list| pd.DataFrame,sb_unit):
    if isinstance(data,list):
        df = pd.DataFrame(data)
    else:
        df = data

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from scipy.optimize import curve_fit 
    def inv_func(x, a, b):
        return a / x + b
    def log_func(x, a, b):
        return a * np.log(x) + b
    try:
        x = STOMATA_DENSITY
        y = STOMATA_SIZE+f'({sb_unit}²)'+'_Mean'
        fig, ax = plt.subplots(figsize = (16,10))
        df_cp = df.dropna(axis=0,subset=['Stomata Density'],inplace=False)
        popt_inv, pcov_inv = curve_fit(inv_func, df_cp[x], df_cp[y])
        popt_log, pcov_log = curve_fit(log_func, df_cp[x], df_cp[y])
        slope, intercept = np.polyfit(df_cp[x], df_cp[y], 1)
        ax = sns.scatterplot(data=df_cp,x=x, y=y)
        x_vals = np.linspace(df_cp[x].min(),df_cp[x].max(),100)
        plt.plot(x_vals, inv_func(x_vals, *popt_inv), color='red', label=f'Fit: a/x + b\n a={popt_inv[0]:.5f}, b={popt_inv[1]:.5f}')
        plt.plot(x_vals, log_func(x_vals, *popt_log), color='green', label=f'Fit: a*log(x) + b\n a={popt_log[0]:.5f}, b={popt_log[1]:.5f}')
        plt.plot(x_vals, slope * x_vals + intercept, color='blue',label=f'Fit: a*x + b\n a={slope:.5f}, b={intercept:.5f}')
        r = df_cp[x].corr(df_cp[y])
        ax.set_xlabel(f'stomata density in mm²')
        ax.set_ylabel(f'stomata size in mm²')
        ax.text(0.05, 0.05, f'n={len(df_cp)}\nr={r:.4f}', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.75))
        plt.legend(loc='upper right',fancybox=True, framealpha=0.5)
        plt.tight_layout()
    except Exception as e:
        fig, ax = plt.subplots(figsize=(16, 10))
    canvas = FigureCanvas(fig)
    canvas.draw()
    plot = Image.frombytes('RGB', canvas.get_width_height(),canvas.tostring_rgb())
    plt.close(fig)
    return plot
def crop(img,bboxes):
    '''
    Crop the given image to different Region of Intersts 
    Return:
    A ndarray of ROIs
    '''
    return [img[y1:y2,x1:x2:,:] for x1,y1,x2,y2 in bboxes.xyxy.numpy().astype(np.int64)]



# Util functions 
def bin_split(list, n):
    '''
        Seperate A List into a Group of Small Subsets Based On n
        Return:
        A List with lists of Elements e.g.: [[a],[b],[d,c]]
    '''
    base = len(list) // n 
    reminder = len(list) % n 
    
    splitted_ls = [list[i*base+ min(i, reminder): (i+1)*base + min(i+1,reminder)] for i in range(n)]
    return splitted_ls
def directory_walk(dir_path):
    '''
        Check The Given Directory Structure, Find directory contains Formatted Image Files,
        Excludes Predict Folder and other Files  
        Return:
        A dict of directory full_path with image count 
        e.g.: {'full_path 1': 3, 'full_path 2': 10, ...}
    '''
    start_time = time.time()
    
    folder_map = {}
    for root, _, files in os.walk(dir_path):
        if 'predict' in Path(root).parts:
            continue # Skip Scan Predict Folder
        if files:
            for f in files: 
                file_path = os.path.join(root,f)
                if f.lower().endswith(IMG_SUFFIXS):
                    if root not in folder_map:
                        folder_map[root] = 1
                    else:
                        folder_map[root]+=1
    end_time =  time.time()
    print(f'{len(folder_map)} folders walk takes {end_time-start_time} seconds')
    return folder_map

# Export
def export_excel(df,dir_path):
    '''
        Export DataFrame to Excel File in the Give Path

        Return:
            Boolean Value Depends on process results
    '''
    try:
        df_copy = df.set_index([FOLDER_ID,IMAGE_ID,STOMATA_ID])
        exl_path = r'results.xlsx'
        df_copy.to_excel(os.path.join(dir_path,exl_path), index = True)
    except Exception as e:
        print(f'An error occurred: {e}', flush=True)
def export_imgs(dest_dir):
    try:
        subdirs = [dir for dir in os.listdir(TEMP_DIR) if os.path.isdir(os.path.join(TEMP_DIR,dir))]
        import shutil
        for dir in subdirs:
            src_full_path = os.path.join(TEMP_DIR,dir)
            dst_full_path = os.path.join(dest_dir,dir)
            shutil.copytree(src_full_path, dst_full_path)
    except Exception as e:
        print(f'An error occurred: {e}', flush=True)

def export_plots(plots: dict | list, dir_path: str):
    try:
        for name,pil in plots.items():
            full_path = os.path.join(dir_path,f'{name}_plot.png') 
            pil.save(full_path)
    except Exception as e:
        print(f'An error occurred: {e}', flush=True)

##-----------------------------------------------------------------------------------
# Verison 1.0
# Add Threading prediction
# Save Images in TEMP to solve Out of Memory when dataset is large
#------------------------------------------------------------------------------------
# Threading Mode
def predict_with_threads_V1_0(model_path,dir_path,run_pore, conf):
    '''
        Start Multiple Model Instances to Conduct Inference in Threads,
        Wait all Threads and Gather the Results 

        Run Single Instance Mode When There Is Only One Directory 

        Determine Number of Threads:
            Min (CPU Cores *4, Number of Sub-Folders)

        Return:
        [Results_directory1, Results_directory2,...]
        And Results itself is a list of result: 
        Results = [r1,r2,r3,....]
    '''
    
    start_time = time.time()
    dirs = [os.path.join(dir_path, sub_path) if sub_path != Path(dir_path).name else dir_path for sub_path in directory_walk(dir_path)]
    if len(dirs) ==1: # Non Thread Mode 
        results= thread_safe_predict_V1_0(model_path, dir_path,run_pore,conf)
        return [results]
    threads_count = min(os.cpu_count() * 4, len(dirs))
    splitted_dirs = bin_split(dirs, threads_count)
    print (f'Run {threads_count} threads and divide into {splitted_dirs}')
    with ThreadPoolExecutor(threads_count) as executor:
        futures = [executor.submit(thread_safe_predict_V1_0,model_path, ls,run_pore,conf) for ls in splitted_dirs]
        merged_results = []
        for future in as_completed(futures):
            merged_results.extend(future.result())
    print(f'{threads_count} threads prediction time : {time.time() - start_time}', flush=True)
    
    return merged_results
# High-Level Threading Inference Prediction (V1.0)
def thread_safe_predict_V1_0(model_path,dir_paths,run_pore,conf):
    ''' 
        Start New Model Instance for Safe Concurrent Interference with Threads 
        The Model Predict All Images in the List

        Return:
            [Results_directory1, Results_directory2,...]
            And Results itself is a list of result: 
            Results = [r1,r2,r3,...]
    '''
    merged_results = []
    # TODO: Not Fully Impleted Yet 
    model_instance = YOLO(model_path)
    for dir_path in dir_paths:
        #start_time = time.time()
        path_name = Path(dir_path).name
        results = model_instance.predict(
                            dir_path,
                            imgsz= 640,
                            conf = conf,
                            save = True,
                            save_crop = run_pore,
                            project = TEMP_DIR,
                            name = path_name,
                            #retina_masks = True # Enhance Mask Quality
                            )
        merged_results.append(results)
        #end_time = time.time()
        #print(f'{path_name} prediction time: {end_time - start_time}')
    return merged_results


##---------------------------------------------------------------------------------------------
# Version 0.1
##-------------------------------------------------------------------------
# Single Model Inference Prediction 
# def start_predict(model_path, dir_path, conf):
#     '''
#         The Single Instance of Model Predicts All Images in Multi-level directory

#         Return:
#         A Collection of Results by Dictionary(k,v) for All Thread Inferences. 
#         Key : Sub-directory Path
#         Value: ultralytics.engine.results.Results 
#     '''
#     start = time.time()
#     model = YOLO(model_path)
#     #predict_path =os.path.join(dir_path,'predict')
#     # if os.path.exists(predict_path):
#     #     shutil.rmtree(predict_path)
    
#     results = model.predict(dir_path,
#                             save = True,
#                             conf= conf,
#                             verbose = True,
#                             project = TEMP_DIR,
#                             name = dir_path
#                             )
#     end = time.time()
#     print(f'Non-threading prediction process took {end - start} seconds')
#     return results


# def split_results(results):
#     '''
#         (Single Model Instance Usage) More Memory!!!
#         Rearrange the results into sub-directory level 

#         Return:
#         A Dictionary (K,V) of Collection of Results,
#         K: Sub-Directory Path 
#         V: A list of Results
#     '''
#     st_time =  time.time()
#     results_dic = {}
#     for r in results:
#         img_path = Path(r.path)
#         parent_path = img_path.parent
#         if parent_path.name not in results_dic:
#             results_dic[parent_path.name] = [r]
#         else:
#             results_dic[parent_path.name].append(r)
#     end_time =  time.time()
#     print(f'split results into {len(results_dic.keys())} folders took {end_time-st_time} seconds')
#     return results_dic
# # END OF V0.1 Methods


#----------------------------------------------
# Analysis Process Based On Model Prediction 
#-----------------------------------------------
# def analysis(results):
#     '''
#         Based on Prediction Results, Store Predicted Image, DataFrame and Plots  

#         Return:
#             Images(Dictionary), Plots(List), Table(DataFrame)
#     '''
#     start_time = time.time()
#     plots = []
#     df = generate_table(results)
#     plots.append(generate_boxplot(df))
#     end_time = time.time()
#     print(f'analysis takes {end_time-start_time}')
#     return plots, df


# def generate_imgs(results):
#         '''
#             Convert Results to PIL Images With Threads 
           
#             Return:
#             A Dictionary Collection of PIL Images 
#         '''
#         if len(results) == 1: # Non Thread Mode
#             return generate_img(results) 
#         #start_time = time.time()
        
#         ## FIXME:Threading Mode Will OOM Since 16GB RAM  
#         # Back To Normal Mode
#         merged_img_dic = {}
#         for rs in results:
#             merged_img_dic = merged_img_dic | generate_img(rs)
#         #print(f'generate images in {len(results)} threads takes: {time.time()-start_time}')
#         return merged_img_dic 


# def generate_table(results):
#     '''
#         Gather Object, Image and Directory Level information 

#         Return: 
#         A Tuple of two lists (DataFrame, DataFrame):
#         Intergreted Information, Directory Only Summary 
#     ''' 
#     data = []
#     img_sum =[]
#     for rs in results:
#         # Image Level Analysis
#         dir_name = None  
#         for r in rs:
#             if dir_name is None:
#                 dir_name = Path(r.path).parent.name
#             detections = r.__len__()
#             if detections != 0:
#                 img_name = Path(r.path).stem
#                 img_area, ratio = img_area_calculator(r.path)
#                 if r.masks is not None: # Segment Model (Stomata Size Feature)
#                     mask_areas = masks_calculator(r.masks,ratio)
#                     for i,size in enumerate(mask_areas):
#                         stomata_info = {
#                             FOLDER_ID: dir_name,
#                             IMAGE_ID: img_name,
#                             STOMATA_ID: int(i),
#                             'Size(mm²)': size
#                         }
#                         data.append(stomata_info)
                    
#                     img_summary = {
#                             FOLDER_ID: dir_name,
#                             IMAGE_ID: f'Summary {img_name}',
#                             STOMATA_ID: None,
#                             'Size(mm²)': np.mean(mask_areas),
#                             'Quatity': detections,
#                             'Density(Quatity)': detections/ img_area,
#                             'Stomata Area': np.sum(mask_areas) / img_area
#                     }
                    
#                 elif r.boxes is not None: # Detection Model
#                     img_summary = {
#                             FOLDER_ID: dir_name,
#                             IMAGE_ID: f'Summary(Image) {img_name}',
#                             STOMATA_ID: None,
#                             'Quatity': detections,
#                             'Size(mm²)': None,
#                             'Density(Quatity)': detections/ img_area,
#                              'Stomata Area': None
#                     }
#                 data.append(img_summary)
#                 img_sum.append(img_summary)
#             else: # No Detection Found
#                 pass
#         df_img_sum = pd.DataFrame(img_sum)
#         dir_summary = {
#             FOLDER_ID: f'Summary {dir_name}',
#             IMAGE_ID: None,
#             STOMATA_ID: None,
#             'Quatity': df_img_sum['Quatity'].mean(),
#             'Quatity SE': df_img_sum['Quatity'].sem(),
#             'Density(Quatity)': df_img_sum['Density(Quatity)'].mean(),
#             'Density(Quatity) SE': df_img_sum['Density(Quatity)'].sem(),
#             'Stomata Area': df_img_sum['Stomata Area'].mean(),
#             'Stomata Area SE': df_img_sum['Stomata Area'].sem()
#         }
#         data.append(dir_summary)
#         #dir_sum.append(dir_summary)
#         df_data = pd.DataFrame(data)
#         # df_data.fillna('',inplace = True)
#        # df_data.set_index([FOLDER_ID,IMAGE_ID, STOMATA_ID], inplace = True)
#     return df_data



