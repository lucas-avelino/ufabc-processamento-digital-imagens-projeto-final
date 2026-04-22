

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

def _buildKernel(size, kernelType):
  if kernelType == KernelType.RECT:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
  elif kernelType == KernelType.ELLIPSE:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
  elif kernelType == KernelType.CROSS:
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
  return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def _watershedMarkers(img, openingSize, sureBgIterations, distanceThreshold, kernelType):
  if img.ndim == 2:
    gray = img.copy()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
  else:
    bgr = img.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

  _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  kernel = _buildKernel(openingSize, kernelType)

  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
  sureBg = cv2.dilate(opening, kernel, iterations=sureBgIterations)

  dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
  _, sureFg = cv2.threshold(dist, distanceThreshold * dist.max(), 255, 0)
  sureFg = np.uint8(sureFg)
  unknown = cv2.subtract(sureBg, sureFg)

  _, markers = cv2.connectedComponents(sureFg)
  markers = markers + 1
  markers[unknown == 255] = 0
  return cv2.watershed(bgr, markers)

def WatershedMask(
  openingSize = 3,
  sureBgIterations = 2,
  distanceThreshold = 0.4,
  kernelType = KernelType.ELLIPSE,
  amountOfAreas = 1,
  sourceIndex = None,
  closeSize = None,
  excludeBorderLabels = True,
  maxAreaRatio = 0.35,
):
  def result(img, previousResults):
    def _fallbackMask(src):
      if src.ndim == 3:
        g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
      else:
        g = src.copy()

      _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      k = _buildKernel(max(3, openingSize), kernelType)
      th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
      th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

      contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if not contours:
        return np.zeros_like(g, dtype=np.uint8)

      h, w = g.shape[:2]
      valid = []
      for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        touchesBorder = x == 0 or y == 0 or (x + cw) >= w - 1 or (y + ch) >= h - 1
        area = cv2.contourArea(cnt)
        if not touchesBorder and area > 0:
          valid.append((area, cnt))

      if not valid:
        return np.zeros_like(g, dtype=np.uint8)

      valid.sort(key=lambda item: item[0], reverse=True)
      out = np.zeros_like(g, dtype=np.uint8)
      cv2.drawContours(out, [valid[0][1]], -1, 255, thickness=-1)
      return out

    source = img
    if sourceIndex is not None and sourceIndex >= 0 and sourceIndex < len(previousResults):
      source = previousResults[sourceIndex]

    markers = _watershedMarkers(source, openingSize, sureBgIterations, distanceThreshold, kernelType)

    validLabels = markers[(markers > 1)]
    if validLabels.size == 0:
      return _fallbackMask(source)

    if excludeBorderLabels:
      borderLabels = set(np.unique(markers[0, :]))
      borderLabels.update(np.unique(markers[-1, :]))
      borderLabels.update(np.unique(markers[:, 0]))
      borderLabels.update(np.unique(markers[:, -1]))
      validLabels = validLabels[~np.isin(validLabels, list(borderLabels))]
      if validLabels.size == 0:
        return _fallbackMask(source)

    uniqueLabels, counts = np.unique(validLabels, return_counts=True)
    if maxAreaRatio is not None:
      maxArea = max(1, int(markers.size * maxAreaRatio))
      allowed = counts <= maxArea
      uniqueLabels = uniqueLabels[allowed]
      counts = counts[allowed]
      if uniqueLabels.size == 0:
        return _fallbackMask(source)

    topCount = max(1, min(amountOfAreas, uniqueLabels.size))
    topIndices = np.argsort(counts)[::-1][:topCount]
    selectedLabels = uniqueLabels[topIndices]

    mask = np.isin(markers, selectedLabels).astype(np.uint8) * 255

    # Close gaps in region borders and keep only external filled components.
    closeKernelSize = closeSize if closeSize is not None else max(3, openingSize * 2 + 1)
    closeKernel = _buildKernel(closeKernelSize, kernelType)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closeKernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filledMask = np.zeros_like(mask)
    if contours:
      cv2.drawContours(filledMask, contours, -1, 255, thickness=-1)
      return filledMask

    return mask
  return result

def Watershed(
  openingSize = 3,
  sureBgIterations = 2,
  distanceThreshold = 0.4,
  kernelType = KernelType.ELLIPSE,
  markBoundaries = True,
  boundaryColor = (0, 0, 255),
  returnMarkers = False,
):
  def result(img, _):
    if img.ndim == 2:
      bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
      gray = img
    else:
      bgr = img.copy()
      gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    markers = _watershedMarkers(img, openingSize, sureBgIterations, distanceThreshold, kernelType)

    if returnMarkers:
      return markers

    if markBoundaries:
      output = bgr.copy()
      output[markers == -1] = boundaryColor
      return output

    labels = markers.copy()
    labels[labels < 0] = 0
    if labels.max() == 0:
      return np.zeros_like(gray)
    labels = np.uint8((labels / labels.max()) * 255)
    return labels
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