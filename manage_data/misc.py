"""
set of functions used for debbuging during development
"""

import numpy as np
import cv2
import json

import os
import os.path as osp
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pylab as plt
import gc

def plot_maps(origin_dir, output_dir):
  files = os.listdir(origin_dir)
  for file in files:
    if file.split('.')[-1] != 'npy':
      continue
    file_name = os.path.join(origin_dir, file)
    print(file_name)
    image = np.load(file_name)
    image = image / np.max(image) * 255
    file_out = os.path.join(output_dir, file.split('.')[0] + '.jpg')
    print(file_out)

    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    plt.imshow(image)
    plt.axis('off')

    fig.savefig(file_out, bbox_inches='tight')
    fig.clf()
    plt.close()
    del a
    gc.collect()

def plot_loss(log_file, out_file, out_options = '1'):
  f = open(log_file, "r")
  loss = []
  mae = []
  mse = []
  for line in f:
    if line.startswith("Epoch:"):
      for item in line.split(","):
        item = item.strip()
        num = float(item.split()[-1])
        if item.startswith("MAE"):
          mae.append(num)
        elif item.startswith("MSE"):
          mse.append(num)
        elif item.startswith("loss:"):
          loss.append(num) 
  assert len(loss) == len(mse) and len(mae) == len(mse), "Error in vector sizes mae: {}, mse: {}, loss: {}".format(len(mae), len(mse), len(loss))
  epoch = np.arange(len(loss))
  if out_options == '1' or out_options == '2':
    plt.plot(epoch, loss, label = 'loss')
  if out_options == '1' or out_options == '3':
    plt.plot(epoch, mae, label = 'mae')
    plt.plot(epoch, mse, label = 'mse')
  plt.xlabel("epoch")
  plt.legend(loc='upper left')
  plt.savefig(out_file)
  plt.clf()
  plt.close()
  gc.collect()

if __name__ == "__main__":
  options = ['1', '2', '3']
  log_file = 'log/QUAD_MSRN_2/l4-level/f16/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold1/log_train.txt'
  out_base_dir = 'log/QUAD_MSRN_2/l4-level/f16/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold1/'
  for opt in options:
    out_file = ''
    if opt == '1':
      out_file = osp.join(out_base_dir, 'log_train_loss_mae_mse.png')
    elif opt == '2':
      out_file = osp.join(out_base_dir, 'log_train_loss.png')
    elif opt == '3':
      out_file = osp.join(out_base_dir, 'log_train_mae_mse.png')
    if out_file != '':
      print("generating plot with option {}".format(opt))
      plot_loss(log_file, out_file, out_options =  opt)
