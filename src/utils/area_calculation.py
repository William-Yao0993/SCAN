import cv2 
import numpy as np
import warnings
import os




#------------------------------------------------------------------------------------------------------------------
# Verison 1.1 
# Mask calculator: return Mask area with mm2, speeding up with torch calculation 
#line detect -> add angle filter 
#------------------------------------------------------------------------------------------------------------------
def img_area(img,ratio):
     if isinstance(img,str):
          img=cv2.imread(img,cv2.IMREAD_COLOR)
     x_in_pxl, y_in_pxl = img.shape[:2]
     x_in_mm = x_in_pxl * ratio
     y_in_mm = y_in_pxl * ratio
     area_in_mm2= x_in_mm * y_in_mm
     return area_in_mm2
def line_detect(img,sb_lenth,default_ratio):
     '''
        Detect and filter the best candidate for scale bar 
        Return:
        if not find -> None 
        success ->  Lines ndarray
     '''
     if isinstance(img,str):
        if os.path.isfile(img):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
     x_start, x_end = int(img.shape[0] * 0.0), int(img.shape[0] * 0.15)  
     y_start, y_end = int(img.shape[1] * 0.60), int(img.shape[1] * 0.73)

     img_roi = img[y_start:y_end,x_start:x_end]
     try:
          
          low_threshold = 45
          high_threahold = int (2.5 * low_threshold)
          edges = cv2.Canny(img_roi, low_threshold,high_threahold)
          
          rho= 1
          theta= np.pi/180
          threshold= 15
          minLineLength=100
          maxLineGap= 30
          min_theta=0
          max_theta=np.pi/2 

          lines = cv2.HoughLinesP(
               edges,
               rho=rho,
               theta=theta,
               threshold=threshold,
               minLineLength=minLineLength,
               maxLineGap=maxLineGap)
          
          if lines is None:
               warnings.warn(f'Fail to detect scale bar, using default ratio: {default_ratio}')
               return default_ratio

          # im_cp = img_roi.copy()
          # for x0,y0,x1,y1 in lines.squeeze():
          #      cv2.line(im_cp,(x0,y0),(x1,y1),(0,0,255),3,0)
          #cv2.imshow('  ',im_cp)        

          # Angle filter
          angles = np.arctan2(np.abs(lines[...,1]-lines[...,3]),np.abs(lines[...,0]-lines[...,2]))
          degrees = angles*180/np.pi
          threshold = 2
          angle_mask = (
               (degrees <=threshold) |
               (np.abs(degrees -90)<= threshold)
          )

          lines = lines[angle_mask]

          x_differences = np.abs(lines[...,0] - lines[...,2])
          y_differences = np.abs(lines[...,1] - lines[...,3])
          x_candidate = np.max(x_differences)
          y_candidate = np.max(y_differences)
          x = lines[x_differences== x_candidate].squeeze()
          y = lines[y_differences== y_candidate].squeeze()

          # cv2.line(img_roi,(x[0],x[1]),(x[2],x[3]),(0,0,255),3,0)
          # cv2.line(img_roi,(y[0],y[1]),(y[2],y[3]),(0,0,255),3,0)
          
          ratio_to_unit = sb_lenth/max(x_candidate,y_candidate)
          # print(ratio_to_unit)
          if np.abs(ratio_to_unit-default_ratio)>2E-5:
               warnings.warn(f'Detected Ratio is much different with default value, using default ratio: {default_ratio}')
               return default_ratio
     except Exception as e :
          print(e)
          warnings.warn(f'Fail to detect scale bar, using default ratio: {default_ratio}')
          return default_ratio
     return ratio_to_unit

def masks_calculator(masks,ratio_to_unit,shape=None):
    '''
        Calculate the Area of each Mask in mm2
        masks -> Results Class (data,orig_shape)
        masks.data -> Torch.tensor with shape N x H x W (H and W could differ with original size)
        Return:
        ndarray with shape N, e.g.   ndarray([s1_in_mm2,s2_in_mm2,...])
    '''
    if masks is None:
         return np.zeros(1) # No Masks Given -> size is 0 
    elif shape is not None:
         orig_x,orig_y = shape
    else:
         orig_x, orig_y = masks.orig_shape
    ratio_x,ratio_y = orig_x/masks.data.shape[1],orig_y/masks.data.shape[2] 
    # print(ratio_x,ratio_y)
    areas_in_pxl = masks.data.numpy().sum(axis=(1,2))
    # print(np.count_nonzero(masks.data,axis=(1,2)))
    areas_in_pxl_orig_shape = areas_in_pxl * ratio_x * ratio_y
    
    # convert areas from original shape pixel to mm2 
    return areas_in_pxl_orig_shape * (ratio_to_unit**2)

#---------------------------------------------------------------------------------------------------------------------------------------------
# Version 1.0 Method
#---------------------------------------------------------------------------------------------------------------------------------------------
# def img_area_calculator_V1_0(img_path): 
#     '''
#         Read the Image and Apply Hough Line Transform to Detect Scale Bar
#         Use Scale Bar Unit as a Reference to Calculate Image Area 
#         from Pixel to Scale Bar Unit and its Ratio 

#         Return:
#         A tuple: (Area,Ratio)
#     '''    
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     lower_threshold = 500
#     threshhold_ratio =3  
#     higher_threshold = int(lower_threshold * threshhold_ratio)
#     edges = cv2.Canny(img, lower_threshold,higher_threshold,apertureSize=5)
#     lines = cv2.HoughLinesP(edges,
#                             rho =1, 
#                             theta=np.pi/180, 
#                             threshold=15, 
#                             minLineLength=100, 
#                             maxLineGap=30)
#     sbs = []
#     x_min, x_max = img.shape[0] *0.0, img.shape[0]*0.1
#     y_min = img.shape[1] * 0.60
#     y_max = img.shape[1] * 0.78
#     # print ('x1,x2: ',x_min, x_max)
#     # print ('y1,y2: ',y_min, y_max)
#     for line in lines:
#         for x1,y1,x2,y2 in line: 
#             if (min(y1,y2) > y_min 
#             and max(y1,y2) < y_max
#             and min(x1,x2) > x_min 
#             and max(x1,x2) < x_max 
            
#             ):
#                     sbs.append(line)
#     # for line in sbs:
#     #     x1,y1,x2,y2 = line[0]
#         #cv2.line(img,(x1,y1), (x2,y2), (0,255,0),2)
#     #print(sbs)

#     # Keep Vertical Scalar Bar Only 
#     if len(sbs) != 1:
#         if len(sbs) == 2: 
#             for sb in sbs:
#                 for x1,y1,x2,y2 in sb:
#                     if np.abs(y2 -y1) < 10:
#                             # Delete Horizontal Scalar Bar
#                             return AERA_DEF, RATIO_DEF
#                             print(f'{img_path} with :{sbs}')
#                             sbs.remove(sb)

#         else:
#              warnings.warn(f"Fail to detect scalar bar, default area and ratio applies {AERA_DEF,RATIO_DEF}")
#              return AERA_DEF, RATIO_DEF
#     #x1,y1,x2,y2 = sbs[0][0]
#     #cv2.line(img,(x1,y1), (x2,y2), (0,255,0),2)
#     # cv2.namedWindow('Scalar Bar Detect', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('Scalar Bar Detect',960,960)
#     #cv2.imwrite(r'c:\Users\u6771897\Desktop\Clear Image\SB Detect.jpg', img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     _ ,y1 ,_ ,y2= sbs[0][0]
#     sb_in_pxl = np.abs(y2-y1)
#     sb_in_mm = 0.05
#     ratio = sb_in_mm / sb_in_pxl

#     x_in_pxl, y_in_pxl = img.shape[:2]
#     x_in_mm = x_in_pxl * ratio
#     y_in_mm = y_in_pxl * ratio
#     area_in_mm2= x_in_mm * y_in_mm
#     if (np.abs(ratio - RATIO_DEF) > 1E-2 or np.abs(area_in_mm2 - AERA_DEF) > 1E-4):
#          warnings.warn(f"Fail to detect scalar bar,using default values {AERA_DEF,RATIO_DEF}")
#          return AERA_DEF, RATIO_DEF
#     return area_in_mm2, ratio

# def masks_calculator_v1_0(masks, ratio):
#     '''
#         Calculate the Area of Each Mask by Given Pixel Ratio

#         Return:
#         A List of Mask Actual Size According to the Ratio  
#     '''
#     areas_in_pxl = []
#     for mask in masks.data:
#         mask_np = mask.numpy().astype(np.uint8)
#         contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:  
#             area_in_pxl = cv2.contourArea(contours[0])
#         else:
#             area_in_pxl = 0
#         areas_in_pxl.append(area_in_pxl)
#     areas_in_mm2 = [i * (ratio **2) for i in areas_in_pxl]
#     return areas_in_mm2

# if __name__ == '__main__':
#     img_area_calculator(r'C:\Users\u6771897\Desktop\Clear Image\00.jpg')