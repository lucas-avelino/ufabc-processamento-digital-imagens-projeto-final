

import cv2
import numpy as np

class KernelType:
  RECT = "RECT"
  ELLIPSE = "ELLIPSE"
  CROSS = "CROSS"

def GaussianBlur(size, sigma):
  def result(img, _):
    return cv2.GaussianBlur(img, (size, size), sigma)
  return result

def MedianBlur(size):
  def result(img, _):
    return cv2.medianBlur(img, size)
  return result

def Sobel(size, scale):
  def result(img, _):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=size, scale=scale)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=size, scale=scale)
    return cv2.convertScaleAbs(grad_x) + cv2.convertScaleAbs(grad_y)
  return result

def EqualizeHist():
  def result(img, _):
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_image[:, :, 0] = cv2.equalizeHist(lab_image[:, :, 0])
    img = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    return img
  return result

def CvtColor(code):
  def result(img, _):
    return cv2.cvtColor(img, code)
  return result

def Erode(size, kernelType = KernelType.RECT):
  def result(img, _):
    if kernelType == KernelType.RECT:
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif kernelType == KernelType.ELLIPSE:
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif kernelType == KernelType.CROSS:
      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    return cv2.erode(img, kernel, iterations=1)
  return result

def Dilate(size, kernelType = KernelType.RECT):
  def result(img, _):
    if kernelType == KernelType.RECT:
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif kernelType == KernelType.ELLIPSE:
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif kernelType == KernelType.CROSS:
      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    return cv2.dilate(img, kernel, iterations=1)
  return result

def Opening(size, kernelType = KernelType.RECT):
  def result(img, _):
    if kernelType == KernelType.RECT:
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif kernelType == KernelType.ELLIPSE:
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif kernelType == KernelType.CROSS:
      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
  return result

def Closing(size, kernelType = KernelType.RECT):
  def result(img, _):
    if kernelType == KernelType.RECT:
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif kernelType == KernelType.ELLIPSE:
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif kernelType == KernelType.CROSS:
      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  return result

def GenerateMask(amountOfAreas = 5):
  def result(img, _):
    if img.ndim == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # images.append(contours)
    mask = np.zeros(img.shape, dtype=np.uint8)
    # get top n contours by area
    topContours = sorted(contours, key=cv2.contourArea, reverse=True)[:amountOfAreas]

    cv2.drawContours(mask, topContours, -1, 255, thickness=-1)
    return mask
  return result

def Parallel(funcs: list):
  def result(img, prevs):
    results = []
    for func in funcs:
      results.append(func(img, prevs))
    return results
  return result

def UsePrevious(index):
  def result(img, previousResults):
    return previousResults[index]
  return result

def PassThrough():
  def result(img, _):
    return img
  return result

def AggregatorYIntoX(f = None, x: int = 0, y: list = [1]):
  def Aggregator(_, previousResults):
    imgs = []
    for i in y:
      branch_output = f(previousResults[-1][i])(previousResults[-1][x], previousResults)
      if isinstance(branch_output, list):
        imgs.extend(branch_output)
      else:
        imgs.append(branch_output)
    return imgs
  return Aggregator

def ParallelAggregable(funcs: list):
  def withMask(mask):
    def result(img, prevs):
      results = []
      for func in funcs:
        results.append(func(mask)(img, prevs))
      return results
    return result
  return withMask

def ApplyHueWithMask(new_hue):
    def withMask(mask):
      def result(img, _):
        h, s, v = cv2.split(img)
        h[mask == 255] = new_hue
        return cv2.merge([h, s, v])
      return result
    return withMask

def ThresholdOtsu():
  def result(img, _):
    if img.ndim == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
  return result

def ResizeKeepAspectRatio(targetW, targetH):
  def result(img, _):
    h, w = img.shape[:2]
    if targetW <= 0 or targetH <= 0 or w == 0 or h == 0:
      return img

    scale = min(targetW / w, targetH / h)
    newW = max(1, int(w * scale))
    newH = max(1, int(h * scale))
    return cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
  return result

def ResizeToOriginal():
  def result(img, imgs):
    return cv2.resize(img, (len(imgs[0][0]), len(imgs[0])), interpolation=cv2.INTER_AREA)
  return result