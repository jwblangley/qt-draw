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
    numpy_image = np.ascontiguousarray(torch_image.cpu().numpy()).astype(np.uint8)

    height, width, channels = numpy_image.shape
    return QtGui.QImage(
        numpy_image.data,
        width,
        height,
        numpy_image.strides[0],
        QtGui.QImage.Format_RGB888,
    )


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.window_size = (500, 500)

        self.img = torch.rand(3, 800, 800)

        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(False)

        self.label = QtWidgets.QLabel()
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )
        self.label.setScaledContents(True)
        self.scrollArea.setWidget(self.label)

        self.canvas = QtGui.QPixmap(self.img.size(2), self.img.size(1))
        self.canvas.fill(QtGui.QColor("white"))
        self.label.setPixmap(self.canvas)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.scrollArea)

        self.last_x = None
        self.last_y = None

        self.paint_image()
        self.label.setPixmap(self.canvas)

        self.label.resize(self.img.size(2), self.img.size(1))
        self.resize(*self.window_size)

    def paint_image(self):
        qImg = torch_to_QImage(self.img)
        img = QImage_to_torch(qImg)
        qImg = torch_to_QImage(img)

        painter = QtGui.QPainter(self.canvas)
        painter.drawImage(0, 0, qImg)

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        painter = QtGui.QPainter(self.canvas)
        painter.setPen(QtCore.Qt.SolidLine)

        pen = painter.pen()
        pen.setWidth(5)
        painter.setPen(pen)

        painter.drawLine(
            self.last_x - self.label.x() - self.scrollArea.x(),
            self.last_y - self.label.y() - self.scrollArea.y(),
            e.x() - self.label.x() - self.scrollArea.x(),
            e.y() - self.label.y() - self.scrollArea.y(),
        )
        painter.end()

        self.label.setPixmap(self.canvas)

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
