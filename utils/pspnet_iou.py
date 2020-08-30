import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from utils.dataloader import DataTransform

class PSPNet_make_mask():
  """
  function: This function can make the inference mask of PSPnet model.
  """

  def __init__(self, img_path_list, anno_path_list, net, input_size, device):
    """
    input
    -------------------------------------
    img_path_list: List of image paths to infer
    anno_path_list: List of anno paths
    net: Pretrained PSPNet model
    input_size: The size of the image to enter into the model
    """
    self.img_path_list = img_path_list
    self.anno_path_list = anno_path_list
    self.net = net.to(device)
    self.device = device
    self.transform = DataTransform(input_size=input_size)
  
  def make_mask(self, idx):
    """
    input
    -------------------------------------
    idx: Index number of the path
    -------------------------------------

    output
    -------------------------------------
    inference_mask: Inference result mask
    annotaion_mask: Grand truth mask
    -------------------------------------
    """
    image_file_path = self.img_path_list[idx]
    anno_file_path = self.anno_path_list[idx]

    img = Image.open(image_file_path)
    img_width, img_height = img.size

    anno_class_img = Image.open(anno_file_path) 
    anno_class_img = anno_class_img.convert('P')
    annotation_mask = anno_class_img
    p_palette = anno_class_img.getpalette()

    img, anno_class_img = self.transform("val", img, anno_class_img)

    self.net.eval()
    x = img.unsqueeze(0)
    x = x.to(self.device, dtype=torch.float)
    outputs = self.net(x)
    y = outputs[0]
    device2 = torch.device('cpu')
    y = y.to(device2)
    y = y[0].detach().numpy()
    y = np.argmax(y, axis=0)

    annotation_mask = np.array(annotation_mask)
    annotation_mask = np.where(annotation_mask==0, 0, 1)
    inference_mask = cv2.resize(np.uint8(y), (img_width, img_height))
    inference_mask = np.where(inference_mask==0, 0, 1)

    return inference_mask, annotation_mask

  def IoU(self, idx, return_mask=False):

    pred_mask, truth_mask = self.make_mask(idx)    
    tp = np.count_nonzero((truth_mask ==1) & (pred_mask == 1))
    tn = np.count_nonzero((truth_mask ==0) & (pred_mask == 0))
    fn = np.count_nonzero((truth_mask ==1) & (pred_mask == 0))
    fp = np.count_nonzero((truth_mask ==0) & (pred_mask == 1))

    if (tp + fp + fn)!=0:
      iou = tp/(tp + fp + fn)
    else:
      None 
      
    if return_mask:
      return pred_mask, truth_mask, iou
    else:
      return iou