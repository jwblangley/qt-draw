import sys
from PySide6 import QtCore, QtWidgets, QtGui

import torch
import numpy as np
import copy


def QImage_to_torch(qimg):
    """
    Convert RGB QImage to CxHxW torch array
    """
    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    width = qimg.width()
    height = qimg.height()

    ptr = qimg.bits()
    np_img = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    np_img = copy.deepcopy(np_img)
    torch_img = torch.Tensor(np_img)
    torch_img = torch_img.permute(2, 0, 1)
    torch_img = torch_img[:-1, :, :]
    torch_img /= 255
    return torch_img


def torch_to_QImage(torch_image):
    """
    Convert CxHxW torch array [0..1] to RGB QImage
    """
    torch_image = copy.deepcopy(torch_image)
    torch_image = torch_image * 255
    torch_image = torch_image.clamp_(0, 255)
    torch_image = torch_image.permute(1, 2, 0)
    ones = torch.ones(torch_image.size(0), torch_image.size(1), 4) * 255
    ones[:, :, :-1] = torch_image
    numpy_image = np.ascontiguousarray(ones.cpu().numpy()).astype(np.uint8)

    height, width, channels = numpy_image.shape
    return QtGui.QImage(
        numpy_image.data,
        width,
        height,
        width * channels,
        QtGui.QImage.Format_RGB32,
    )


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.window_size = (600, 600)

        self.img = torch.rand(3, 400, 400)
        self.scale = 1

        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(False)
        self.scrollArea.setAlignment(QtCore.Qt.AlignCenter)
        self.scrollArea.setStyleSheet("QScrollArea {background-color: #303030}")

        self.label = QtWidgets.QLabel()
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )
        self.label.setScaledContents(True)
        self.scrollArea.setWidget(self.label)

        self.canvas = QtGui.QPixmap(self.img.size(2), self.img.size(1))
        self.canvas.fill(QtGui.QColor("white"))

        self.painter = QtGui.QPainter(self.canvas)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setSpacing(10)
        self.layout.addWidget(self.scrollArea, 0, 0)

        self.shortcut_zoom_in = QtGui.QShortcut(QtGui.QKeySequence.ZoomIn, self)
        self.shortcut_zoom_in.activated.connect(lambda: self.zoom_label(factor=2))

        self.shortcut_zoom_out = QtGui.QShortcut(QtGui.QKeySequence.ZoomOut, self)
        self.shortcut_zoom_out.activated.connect(lambda: self.zoom_label(factor=0.5))

        self.last_x = None
        self.last_y = None

        self.mask = torch.zeros_like(self.img, dtype=torch.bool)

        self.paint_image()

        self.zoom_label()
        self.resize(*self.window_size)

    def zoom_label(self, factor=None):
        if factor is not None:
            self.scale *= factor
        self.label.resize(self.img.size(2) * self.scale, self.img.size(1) * self.scale)

    def paint_image(self):
        qImg = torch_to_QImage(self.img)

        self.painter.drawImage(0, 0, qImg)
        self.label.setPixmap(self.canvas)

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        self.painter.setPen(QtCore.Qt.SolidLine)

        pen = self.painter.pen()
        pen.setWidth(5)
        self.painter.setPen(pen)

        self.painter.drawLine(
            (self.last_x - self.label.x() - self.scrollArea.x()) / self.scale,
            (self.last_y - self.label.y() - self.scrollArea.y()) / self.scale,
            (e.x() - self.label.x() - self.scrollArea.x()) / self.scale,
            (e.y() - self.label.y() - self.scrollArea.y()) / self.scale,
        )

        self.label.setPixmap(self.canvas)

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

        self.switch_background()

    def switch_background(self):
        current = QImage_to_torch(self.canvas.toImage())
        self.mask |= ~torch.isclose(current, self.img, atol=1 / 255)

        new_img = torch.rand(*self.img.shape)
        new_img[self.mask] = current[self.mask]
        self.img = new_img
        self.paint_image()


app = QtWidgets.QApplication([])
window = MainWindow()
window.show()
app.exec_()
