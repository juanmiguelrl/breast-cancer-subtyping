import numpy as np
import skimage.morphology as sk_morphology
import scipy.ndimage.morphology as sc_morph
import math
import skimage.feature as sk_feature

import glob
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000


def mask_percent(np_img):
    #it determines and returns which percentage of the input image is masked by counting how many values are 0
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage

def filter_binary_dilation(np_img, disk_size=1, iterations=1, output_type="uint8"):
    #if performs a dilation to the binary image passed
    #the disk size indicate the radius of the disk kernel used in the dilation process
    # iterations indicate how many times repeat the process
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_dilation(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result

def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8"):
    #it closes a binary image applying a dilation followed by an erosion
    #it can be used in order to remove small holes in the image
    #the disk size indicate the radius of the disk kernel used in the closing process
    #iterations indicate how many times repeat the process
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result

def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
    #it removes the small objects from the image which are of less size than the indicated in min_size
    #if avoid_overmask is true the function call itself recursively with smaller minimum size to remove reduce
    #the masking performed by the filter
    #if avoid_overmask is true it avoids masking above the thresh percentage indicated in overmask_thresh

  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
      mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  return np_img

def filter_binary_fill_holes(np_img, output_type="bool"):
    #it fill the holes in a binary image
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_fill_holes(np_img)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
    #it makes a mask to filter values over a green channel threshold, as hematoxylin and eosin are of
    #purple and pink colors, and these colors do not have too much green
    #avoid_overmask is a boolean that if is true it avoids masking above the overmask_thresh percentage indicated

  g = np_img[:, :, 1]
  gr_ch_mask = (g < green_thresh) & (g > 0)
  mask_percentage = mask_percent(gr_ch_mask)
  if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
    new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
    print(
      "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
        mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
    gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
  np_img = gr_ch_mask

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  return np_img


def filter_grays(rgb, tolerance=15, output_type="bool"):
    # it creates a mask which filters the gray colors, which are the ones that
    #have similar values of red, green and blue channels
    #the tolerance determines how similiar the color values must be to be filtered

  (h, w, c) = rgb.shape

  rgb = rgb.astype(np.int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool"):
    # it creates a mask which filters the red colors based on a pixel value below the blue and green channel
    # threshold values and above the red channel threshold value
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result

def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool"):
    #it creates a mask which filters the blue colors based on a pixel value below the red and green channel
    #threshold values and above the blue channel threshold value
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool"):
    # it creates a mask which filters the green colors based on a pixel value above the green and blue channel
    # threshold values and below the red channel threshold value
    #as the blue and green channel tend to track together, a blue channel lower threshold instead of an upper is used
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255

  return result

def filter_red_pen(rgb, output_type="bool"):
    #it returns a mask filtering the red pen in the slides
  result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
           filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result

def filter_green_pen(rgb, output_type="bool"):
    # it returns a mask filtering the green pen in the slides
  result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255

  return result


def filter_blue_pen(rgb, output_type="bool"):
    # it returns a mask filtering the blue pen in the slides
  result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255

  return result

def filter_rgb_to_grayscale(np_img, output_type="uint8"):
    #to pass the image from rgb to grayscale
  grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])

  if output_type != "float":
    grayscale = grayscale.astype("uint8")

  return grayscale

def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"):
  #sigma is the width (std dev) of Gaussian
  #low_threshold is the Low hysteresis threshold value
  #high_threshold is the High hysteresis threshold value
  can = sk_feature.canny(np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    can = can.astype(float)
  else:
    can = can.astype("uint8") * 255
  return can

###############################################
from skimage.measure import label
def getLargestCC(segmentation):
    #used to get the largest connected component from the image
  labels = label(segmentation)
  assert (labels.max() != 0)  # assume at least 1 CC
  largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
  return largestCC

###############################################

def is_empty_area(result,mask,threshold=0.5):
  #checks if the gray area is a big part of the image by comparing the area of the image and the area of the gray area
  gray_mask = filter_grays(result)
  gray_area = gray_mask.sum()
  img_area = mask.sum()
  gray_area_ratio = gray_area/img_area
  if gray_area_ratio < threshold:
    return True,gray_area_ratio
  else:
    return False,gray_area_ratio




def mask(img,remove_blue,remove_red,remove_green,only_one,empty_threshold,tissue_closing,remove_small_size,fill_holes):
    #it applies the different masks indicated by the used to the image
    gray_mask = filter_grays(img)
    #red_mask = filter_red(img, red_lower_thresh = 180, green_upper_thresh=80, blue_upper_thresh=90)
    red_mask = filter_red_pen(img)
    green_mask = filter_green_pen(img)
    blue_mask = filter_blue_pen(img)
    color_mask = green_mask & gray_mask & red_mask & blue_mask

    #mask = img.copy()
    mask = filter_binary_dilation(color_mask,output_type="bool")
    mask = filter_binary_closing(mask, tissue_closing,output_type="bool")
    mask = filter_remove_small_objects(mask, remove_small_size,output_type="bool")
    if fill_holes:
        mask = filter_binary_fill_holes(mask)


    #result = img * np.dstack([mask, mask, mask])
    print("the percentage of the image that is masked is: ", mask_percent(mask))

    if only_one:
        mask = getLargestCC(mask)
    if remove_blue:
        mask = mask & blue_mask
    if remove_red:
        mask = mask & red_mask
    if remove_green:
        mask = mask & green_mask
    result = img * np.dstack([mask, mask, mask])
    print("the percentage of the biggest connected component that is masked is: ", mask_percent(mask))
    check,ratio = is_empty_area(result, mask,empty_threshold)
    if check:
      print("the area of the masked area is empty: ", ratio)
      return result,False
    else:
      print("the area of the masked area is not empty: ", ratio)
      return result, True

###############
def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string
###############
def remove_prefix(text, prefix):
  if text.startswith(prefix):
    return text[len(prefix):]
  return text
###############


def get_filename(path):
    return path.split('\\')[-1]

def crop_resize_image(original_image,resize_size):
    cropped_image = original_image.crop(original_image.getbbox())
    # resize the image
    resized_image = cropped_image.resize((resize_size,resize_size))
    # plot image
    # plt.imshow(resized_image)
    # plt.show()
    # save image
    return resized_image
################################################

def mask_list(input_dir,destination_path,resize_size,only_tissue,tissue_closing,remove_small_size,fill_holes,canny,discard,crop,resize,
              remove_blue,remove_red,remove_green,only_one,empty_threshold,canny_params):
    for img in glob.glob(os.path.join(input_dir,"*")):
      #check if is an file
      if os.path.isfile(img):
        np_img = np.asarray(Image.open(img).convert('RGB'))
        if canny:
          canny_result = filter_canny(filter_rgb_to_grayscale(np_img),canny_params["sigma"],canny_params["low_threshold"],canny_params["high_threshold"],output_type="bool")
          np_img = np_img * np.dstack([~canny_result, ~canny_result, ~canny_result])
          check = True
        if only_tissue:
          result,check = mask(np_img,remove_blue,remove_red,remove_green,only_one,empty_threshold,tissue_closing,remove_small_size,fill_holes)
          np_img = result
        elif not canny:
            check = True
            np_img = np.asarray(Image.open(img))
        np_img = Image.fromarray(np_img)
        if crop:
          np_img = np_img.crop(np_img.getbbox())
        if resize:
          np_img = np_img.resize(resize_size)
        if np_img.mode == 'L':
          np_img = np_img.convert('RGB')
        if check or not discard:
          np_img.save(os.path.join(destination_path,os.path.basename(img)))
        else:
          np_img.save( os.path.join(os.path.join(destination_path,"discarded"), os.path.basename(img)))

def filter_images(input_dir,destination_path,resize_size=(896,896),only_tissue=True,tissue_closing=10,remove_small_size=100000,fill_holes=True,canny=False,discard=True,crop=True,
                  resize=True,remove_blue=False,remove_red=False,remove_green=False,only_one=True,empty_threshold=0.5,
                  canny_params={}):
  if not os.path.exists(remove_prefix(destination_path,"/")):
      os.makedirs(remove_prefix(destination_path,"/"))

  if not os.path.exists(destination_path+"/discarded"):
      os.makedirs(destination_path+"/discarded")

  mask_list(input_dir,destination_path,resize_size,only_tissue,tissue_closing,remove_small_size,fill_holes,canny,discard,crop,resize,remove_blue,remove_red,remove_green,
            only_one,empty_threshold,canny_params)

def filter_images_multiple(list_of_dictionaries):
    for PARAMS in list_of_dictionaries:
        filter = {"resize_size": (896,896) ,"only_tissue":True,"tissue_closing":10, "remove_small_size":100000,"fill_holes":True,
                  "canny":False,"discard":True,"crop":True,
                  "resize":True,"remove_blue_pen":False,"remove_red_pen":False,"remove_green_pen":False,"only_one_tissue":True,"empty_threshold":0.5,
                  "canny_params":{"sigma":1.0,"low_threshold":0,"high_threshold":25}}
        filter.update(PARAMS)
        filter_images(filter["input_dir"], filter["destination_path"], filter["resize_size"],
                              filter["only_tissue"],filter["tissue_closing"], filter["remove_small_size"],filter["fill_holes"]
                                ,filter["canny"],filter["discard"],filter["crop"],
                              filter["resize"],filter["remove_blue_pen"],filter["remove_red_pen"],filter["remove_green_pen"],
                              filter["only_one_tissue"],filter["empty_threshold"],filter["canny_params"])
