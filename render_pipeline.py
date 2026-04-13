import cv2
import numpy as np
import matplotlib.pyplot as plt
import types

def _iter_display_items(step, label_prefix):
  if isinstance(step, list):
    for idx, item in enumerate(step, start=1):
      yield from _iter_display_items(item, f"{label_prefix}.{idx}")
  else:
    yield label_prefix, step

def _pipeline_display_items(steps):
  items = []
  for step_idx, step in enumerate(steps, start=1):
    items.extend(_iter_display_items(step, f"Step {step_idx}"))
  return items

def RunPipeline(img, pipeline):
  steps = [img]

  for command in pipeline:
    previous = steps[-1]

    if isinstance(previous, list):
      if isinstance(command, list):
        if len(command) != len(previous):
          raise ValueError(
            f"Length of command list ({len(command)}) must match number of parallel branches ({len(previous)})."
          )
        current = [cmd(branch, steps) for cmd, branch in zip(command, previous)]
      elif isinstance(command, types.FunctionType) and "Aggregator" in command.__qualname__:
        current = command(None, steps)
      else:
        current = [command(branch, steps) for branch in previous]
      steps.append(current)
    else:
      steps.append(command(previous, steps))

  return steps

def ShowAll(images, itemPerRow=3):
  if not images:
    return

  items = _pipeline_display_items(images)
  if not items:
    return

  cols = min(itemPerRow, len(items))
  rows = (len(items) + cols - 1) // cols

  fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

  if rows == 1 and cols == 1:
    axes = [[axes]]
  elif rows == 1:
    axes = [axes]
  elif cols == 1:
    axes = [[ax] for ax in axes]

  for i in range(rows * cols):
    r, c = divmod(i, cols)
    ax = axes[r][c]
    if i < len(items):
      title, img = items[i]
      if len(img.shape) == 2:
        ax.imshow(img, cmap="gray")
      else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      ax.set_title(title)
    ax.axis("off")

  fig.tight_layout()
  plt.show()

def RunCameraPipeline(
  pipeline,
  cameraIndex=0,
  itemPerRow=3,
  tileWidth=360,
  windowName="Camera Pipeline",
  resizableWindow=True,
  fitToWindow=True,
  minTileWidth=180,
  initialWindowSize=(1280, 720),
  gcEveryNFrames=60
):
  import gc

  def _letterbox_to_window(img, targetW, targetH):
    h, w = img.shape[:2]
    if targetW <= 0 or targetH <= 0 or w == 0 or h == 0:
      return img

    scale = min(targetW / w, targetH / h)
    newW = max(1, int(w * scale))
    newH = max(1, int(h * scale))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((targetH, targetW, 3), dtype=np.uint8)
    offX = (targetW - newW) // 2
    offY = (targetH - newH) // 2
    canvas[offY:offY + newH, offX:offX + newW] = resized
    return canvas

  windowFlags = cv2.WINDOW_NORMAL if resizableWindow else cv2.WINDOW_AUTOSIZE
  if resizableWindow and hasattr(cv2, "WINDOW_KEEPRATIO"):
    windowFlags |= cv2.WINDOW_KEEPRATIO

  cv2.namedWindow(windowName, windowFlags)
  if resizableWindow and initialWindowSize is not None:
    cv2.resizeWindow(windowName, initialWindowSize[0], initialWindowSize[1])

  cap = cv2.VideoCapture(cameraIndex)
  if not cap.isOpened():
    cv2.destroyWindow(windowName)
    raise RuntimeError(f"Could not open camera index {cameraIndex}")

  frameCount = 0

  try:
    while True:
      ok, frame = cap.read()
      if not ok:
        break

      steps = RunPipeline(frame, pipeline)
      items = _pipeline_display_items(steps)
      if items:
        cols = min(itemPerRow, len(items))
        targetTileWidth = max(minTileWidth, tileWidth)

        tiles = []
        for title, img in items:
          if img.ndim == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
          else:
            vis = img.copy()

          h, w = vis.shape[:2]
          tileHeight = max(1, int(h * (targetTileWidth / w)))
          vis = cv2.resize(vis, (targetTileWidth, tileHeight), interpolation=cv2.INTER_AREA)
          cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
          tiles.append(vis)

        rows = (len(tiles) + cols - 1) // cols
        blank = np.zeros_like(tiles[0])
        gridRows = []
        for r in range(rows):
          rowTiles = []
          for c in range(cols):
            idx = r * cols + c
            rowTiles.append(tiles[idx] if idx < len(tiles) else blank)
          gridRows.append(np.hstack(rowTiles))

        grid = np.vstack(gridRows)

        if resizableWindow and fitToWindow and hasattr(cv2, "getWindowImageRect"):
          try:
            _, _, windowW, windowH = cv2.getWindowImageRect(windowName)
            if windowW > 0 and windowH > 0:
              grid = _letterbox_to_window(grid, windowW, windowH)
          except cv2.error:
            pass

        cv2.imshow(windowName, grid)

      frameCount += 1
      if frameCount % max(1, gcEveryNFrames) == 0:
        gc.collect()

      key = cv2.waitKey(1) & 0xFF
      if key in (27, ord("q")):  # ESC or q
        break
  finally:
    cap.release()
    cv2.destroyWindow(windowName)
    gc.collect()

def _render_pipeline_grid(
  frame,
  pipeline,
  itemPerRow=3,
  tileWidth=360,
  minTileWidth=180
):
  steps = RunPipeline(frame, pipeline)
  items = _pipeline_display_items(steps)
  if not items:
    return None

  cols = min(itemPerRow, len(items))
  targetTileWidth = max(minTileWidth, tileWidth)

  tiles = []
  for title, img in items:
    if img.ndim == 2:
      vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
      vis = img.copy()

    h, w = vis.shape[:2]
    tileHeight = max(1, int(h * (targetTileWidth / w)))
    vis = cv2.resize(vis, (targetTileWidth, tileHeight), interpolation=cv2.INTER_AREA)
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    tiles.append(vis)

  rows = (len(tiles) + cols - 1) // cols
  blank = np.zeros_like(tiles[0])
  gridRows = []
  for r in range(rows):
    rowTiles = []
    for c in range(cols):
      idx = r * cols + c
      rowTiles.append(tiles[idx] if idx < len(tiles) else blank)
    gridRows.append(np.hstack(rowTiles))

  return np.vstack(gridRows)

def RunCameraPipelineWithSliders(
  pipelineBuilder,
  sliders,
  cameraIndex=0,
  itemPerRow=3,
  tileWidth=360,
  minTileWidth=180,
  windowName="Camera",
  controlsWindowName="Parametros",
  resizableWindow=True,
  fitToWindow=True,
  initialWindowSize=(1280, 720)
):
  try:
    import importlib

    QtCore = importlib.import_module("PyQt5.QtCore")
    QtGui = importlib.import_module("PyQt5.QtGui")
    QtWidgets = importlib.import_module("PyQt5.QtWidgets")

    Qt = QtCore.Qt
    QTimer = QtCore.QTimer
    QImage = QtGui.QImage
    QPixmap = QtGui.QPixmap
    QApplication = QtWidgets.QApplication
    QHBoxLayout = QtWidgets.QHBoxLayout
    QLabel = QtWidgets.QLabel
    QMainWindow = QtWidgets.QMainWindow
    QScrollArea = QtWidgets.QScrollArea
    QSlider = QtWidgets.QSlider
    QVBoxLayout = QtWidgets.QVBoxLayout
    QWidget = QtWidgets.QWidget
  except ImportError as exc:
    raise RuntimeError(
      "PyQt5 is required for RunCameraPipelineWithSliders. Install with: pip install pyqt5"
    ) from exc

  def _clamp_slider_value(minValue, maxValue, value, odd=False):
    value = max(minValue, min(maxValue, int(value)))
    if odd and value % 2 == 0:
      value += 1
      if value > maxValue:
        value = max(minValue, value - 2)
    return value

  if not isinstance(sliders, list) or len(sliders) == 0:
    raise ValueError("sliders must be a non-empty list of slider specifications")

  sliderSpecs = []
  for slider in sliders:
    if "name" not in slider or "min" not in slider or "max" not in slider:
      raise ValueError("Each slider must have keys: name, min, max")
    if slider["max"] < slider["min"]:
      raise ValueError(f"Slider {slider['name']} has max < min")

    name = str(slider["name"])
    label = str(slider.get("label", name))
    minValue = int(slider["min"])
    maxValue = int(slider["max"])
    odd = bool(slider.get("odd", False))
    initial = _clamp_slider_value(minValue, maxValue, slider.get("initial", minValue), odd)
    sliderSpecs.append(
      {
        "name": name,
        "label": label,
        "min": minValue,
        "max": maxValue,
        "initial": initial,
        "odd": odd,
      }
    )

  class _PipelineQtWindow(QMainWindow):
    def __init__(self):
      super().__init__()
      self.setWindowTitle(windowName)
      if initialWindowSize is not None:
        self.resize(initialWindowSize[0], initialWindowSize[1])

      self.cap = cv2.VideoCapture(cameraIndex)
      if not self.cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cameraIndex}")

      central = QWidget()
      self.setCentralWidget(central)
      rootLayout = QHBoxLayout(central)

      controlsContainer = QWidget()
      controlsLayout = QVBoxLayout(controlsContainer)
      controlsLayout.setContentsMargins(8, 8, 8, 8)
      controlsLayout.setSpacing(10)

      titleLabel = QLabel(controlsWindowName)
      titleLabel.setStyleSheet("font-size: 18px; font-weight: 600;")
      controlsLayout.addWidget(titleLabel)

      self.sliderWidgets = []
      for spec in sliderSpecs:
        rowTitle = QLabel(spec["label"])
        rowTitle.setWordWrap(True)
        rowTitle.setStyleSheet("font-size: 13px; font-weight: 500;")

        row = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(spec["min"])
        slider.setMaximum(spec["max"])
        slider.setValue(spec["initial"])
        valueLabel = QLabel(str(spec["initial"]))
        valueLabel.setMinimumWidth(58)
        valueLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row.addWidget(slider, 1)
        row.addWidget(valueLabel)

        controlsLayout.addWidget(rowTitle)
        controlsLayout.addLayout(row)

        self.sliderWidgets.append({"spec": spec, "slider": slider, "valueLabel": valueLabel})

      controlsLayout.addStretch(1)

      scroll = QScrollArea()
      scroll.setWidgetResizable(True)
      scroll.setWidget(controlsContainer)
      scroll.setMinimumWidth(420)
      scroll.setMaximumWidth(560)

      self.previewLabel = QLabel()
      self.previewLabel.setAlignment(Qt.AlignCenter)
      self.previewLabel.setMinimumSize(640, 360)

      rootLayout.addWidget(scroll, 0)
      rootLayout.addWidget(self.previewLabel, 1)

      self.timer = QTimer(self)
      self.timer.timeout.connect(self._update_frame)
      self.timer.start(16)

    def _current_params(self):
      params = {}
      for item in self.sliderWidgets:
        spec = item["spec"]
        rawValue = item["slider"].value()
        value = _clamp_slider_value(spec["min"], spec["max"], rawValue, spec["odd"])
        if value != rawValue:
          item["slider"].blockSignals(True)
          item["slider"].setValue(value)
          item["slider"].blockSignals(False)
        item["valueLabel"].setText(str(value))
        params[spec["name"]] = value
      return params

    def _update_frame(self):
      ok, frame = self.cap.read()
      if not ok:
        return

      params = self._current_params()

      try:
        pipeline = pipelineBuilder(params)
        grid = _render_pipeline_grid(
          frame,
          pipeline,
          itemPerRow=itemPerRow,
          tileWidth=tileWidth,
          minTileWidth=minTileWidth,
        )
      except Exception:
        return

      output = grid
      if output is None:
        return

      rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
      h, w = rgb.shape[:2]
      qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
      pixmap = QPixmap.fromImage(qimg)

      if fitToWindow:
        pixmap = pixmap.scaled(
          self.previewLabel.size(),
          Qt.KeepAspectRatio,
          Qt.SmoothTransformation,
        )

      self.previewLabel.setPixmap(pixmap)

    def closeEvent(self, event):
      self.timer.stop()
      if self.cap is not None and self.cap.isOpened():
        self.cap.release()
      super().closeEvent(event)

  app = QApplication.instance()
  ownsApp = False
  if app is None:
    app = QApplication([])
    ownsApp = True

  window = _PipelineQtWindow()
  window.show()

  if ownsApp:
    app.exec_()
