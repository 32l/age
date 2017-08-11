from __future__ import division
import cv2
import numpy as np

# basic image processing interface based on openCV
def imread(filename):
    return cv2.imread(filename)

def imwrite(im, filename):
    cv2.imwrite(filename, im)
    return

def resize(im,  new_shape):
    ''' im_out = resize(im_in, new_shape = (w, h)) '''
    
    new_shape = list(new_shape)
    assert(new_shape[0] > 0 or new_shape[1] > 0)
    if new_shape[0] < 0:
        new_shape[0] = int(new_shape[1] / im.shape[0] * im.shape[1])
    elif new_shape[1] < 0:
        new_shape[1] = int(new_shape[0] / im.shape[1] * im.shape[0])

    return cv2.resize(im, tuple(new_shape), interpolation = cv2.INTER_LINEAR)
    
def crop(im, bbox):
    ''' im_patch = crop(im, bbox = (x1, y1, x2, y2))'''
    x1, y1, x2, y2 = bbox
    h, w = im.shape[0:2]
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(max(min(x2, w-1), x1))
    y2 = int(max(min(y2, h-1), y1))

    return im[y1:(y2+1), x1:(x2+1), :].copy()

def upsample(im, size, fill = True):
    '''
    up sample the image to have size as its long-side length
    if fill == ture, the up sampled image will be padded to size x size with 0. 
    '''
    new_shape = list(im.shape[0:2])
    if new_shape[0]>new_shape[1]:
        new_shape[0:2] = [size, new_shape[1] / new_shape[0] * size]
    else:
        new_shape[0:2] = [new_shape[0]/new_shape[1] * size, size]

    new_shape = np.array(new_shape, dtype = np.int)
    if fill == True:
        im_out = np.zeros((size, size, 3), dtype = im.dtype)
        im_out[0:new_shape[0], 0:new_shape[1], :] = resize(im, new_shape[::-1])
    else:
        im_out = resize(im, new_shape[::-1])

    return im_out

color_map = {'r':(255,0,0), 'g':(0,255,0), 'b':(0,0,255),\
             'y':(255,255,0), 'p':(255,0,255), 'c':(0,255,255),\
             'rg':(255,255,0), 'rb':(255,0,255), 'gb':(0,255,255),\
             'w':(255,255,255), 'k':(0,0,0), 'wk':(100,100,100)}
def _color_transfer(color):
    if isinstance(color, tuple):
        return color
    else:
        assert(isinstance(color, str))
        if color in color_map:
            return color_map[color]
        else:
            return (0,0,0)

def draw_rectangle(im, position, color, thickness = 1):
    ''' draw_rectangle(im, position = (x1, y1, x2, y2), color = (r, g, b))
        (x1, y1), (x2, y2) are left-top and right-bottom point'''
    color = _color_transfer(color)
    position = np.array(position, dtype=int)
    cv2.rectangle(im, tuple(position[0:2]), tuple(position[2:4]), tuple(color[::-1]), thickness)
    return im

def draw_text(im, str_text, position, color, font_size = 1):
    ''' draw_text(im, str_text, position = (x, y), color = (r, g, b), font_size = 1.0)
        (x, y) is bottom-left point '''
    color = _color_transfer(color)
    position = tuple(np.array(position, dtype = np.int))
    cv2.putText(im, str_text.encode('ascii', 'replace'), position, cv2.FONT_HERSHEY_SIMPLEX, font_size, tuple(color[::-1]))
    return im

def imshow(im, window_name = 'default'):
    cv2.imshow(window_name, im)
    key = cv2.waitKey(0)
    # print(key)
    cv2.destroyWindow(window_name)
    if key ==27:
        raise Exception('Esc pressed!')
    return

def add_tag(im, tag_list, color_list = None):
    if color_list == None or len(color_list) != len(tag_list):
        color_list = [(0,0,0) for i in xrange(len(tag_list))]
    else:
        color_list = [_color_transfer(c) for c in color_list]

    height, width = im.shape[0], im.shape[1]
    # new_height = max(height, len(tag_list) * 20)
    # new_width = width + max(200, max([len(t) for t in tag_list]) * 10 + 10)
    # new_im = np.zeros((new_height, new_width, 3), dtype = im.dtype)
    # new_im[0:height, 0:width, :] = im
    # for i, tag in enumerate(tag_list):
    #   draw_text(new_im, tag, (width + 10, i * 20 + 20), (0,0,0), font_size = 1)
    new_height = height + 20 * len(tag_list) + 10
    new_width = max(width, max([len(t) for t in tag_list])* 10 + 10)
    new_im = np.ones((new_height, new_width, 3), dtype = im.dtype) * 255
    new_im[0:height, 0:width, :] = im
    for i, tag in enumerate(tag_list):
        draw_text(new_im, tag, (10, height + (i+1) * 20), color = color_list[i], font_size = 0.5)
    return new_im

# all boxes are [xmin, ymin, xmax, ymax] format, 0-indexed, including xmax and ymax
def compute_iou(boxes, target):
    assert(target.ndim == 1 and boxes.ndim == 2)
    A_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    A_target = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)
    assert(np.all(A_boxes >= 0))
    assert(np.all(A_target >= 0))
    I_x1 = np.maximum(boxes[:, 0], target[0])
    I_y1 = np.maximum(boxes[:, 1], target[1])
    I_x2 = np.minimum(boxes[:, 2], target[2])
    I_y2 = np.minimum(boxes[:, 3], target[3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_boxes + A_target - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

def stitch(image_list, axis):
    '''
    image_list: a list of images (3-d numpy.ndarray)
    axis: 0 for horizontal and 1 for vertical
    '''
    if axis == 0:
        w = sum([im.shape[1] for im in image_list])
        h = max([im.shape[0] for im in image_list])
        new_im = np.zeros((h, w, 3), dtype = np.float32)
        n = 0
        for im in image_list:
            new_im[0:im.shape[0],n:(n+im.shape[1]),:] = im
            n += im.shape[1]
    else:
        w = max([im.shape[1] for im in image_list])
        h = sum([im.shape[0] for im in image_list])
        new_im = np.zeros((h, w, 3), dtype = np.float32)
        n = 0
        for im in image_list:
            new_im[n:(n+im.shape[0]),0:im.shape[1],:] = im
            n += im.shape[0]
    return new_im

def bgr2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def combine_bbox(bbox_list):
    bbox_mat = np.array(bbox_list, dtype = np.float32)
    bbox = np.min(bbox_mat, axis = 0)
    bbox[2:4] = np.max(bbox_mat[:,2:4], axis = 0)
    return bbox.tolist()