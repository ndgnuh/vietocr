import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog

from vietocr.dataloaders import Sample, get_dataset


def numpy_to_pixmap(image: np.ndarray):
    h, w = image.shape[:2]
    bytes_per_line = w * image.shape[-1]
    fmt = QtGui.QImage.Format.Format_BGR888
    qimage = QImage(image.tobytes(), w, h, bytes_per_line, fmt)
    pixmap = QPixmap(qimage)
    return pixmap


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data_path=None):
        super().__init__()
        self.setupUi()
        self.action_open_dataset.triggered.connect(self.on_open_dataset)
        self.action_next.triggered.connect(self.next_index)
        self.action_prev.triggered.connect(self.prev_index)

        self.data_path = data_path
        if data_path is not None:
            self.dataset = get_dataset(data_path, transform=None)
            self.switch_to_data(0)
        else:
            self.dataset = None
            self.data_index = 0

    def next_index(self):
        if self.dataset is None:
            return
        index = self.data_index + 1
        index = min(index, len(self.dataset) - 1)
        self.switch_to_data(index)

    def prev_index(self):
        if self.dataset is None:
            return
        index = self.data_index - 1
        index = max(index, 0)
        self.switch_to_data(index)

    def switch_to_data(self, index):
        if self.dataset is None:
            return
        self.data_index = index
        sample = self.dataset[self.data_index]
        image = numpy_to_pixmap(np.array(sample.image))
        label = "".join(sample.target)

        total = len(self.dataset)
        msg = f"[{self.data_index}/{total}] {self.data_path}"
        self.statusBar().showMessage(msg)
        self.image.setPixmap(image)
        self.target.setText(label)

    def on_open_dataset(self):
        file, _ = QFileDialog.getOpenFileName(self)
        if file == "":
            return

        self.data_path = file
        self.data_index = 0
        self.dataset = get_dataset(file, transform=None)

        self.switch_to_data(0)

    def setupUi(self):
        MainWindow = self
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.image = QtWidgets.QLabel(parent=self.centralwidget)
        self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image.setObjectName("image")
        self.verticalLayout.addWidget(self.image)
        self.target = QtWidgets.QLabel(parent=self.centralwidget)
        self.target.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.target.setObjectName("target")
        self.verticalLayout.addWidget(self.target)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 36))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_open_dataset = QtGui.QAction(parent=MainWindow)
        self.action_open_dataset.setObjectName("action_open_dataset")
        self.action_next = QtGui.QAction(parent=MainWindow)
        self.action_next.setObjectName("action_next")
        self.action_prev = QtGui.QAction(parent=MainWindow)
        self.action_prev.setObjectName("action_prev")
        self.menuFile.addAction(self.action_open_dataset)
        self.menuFile.addAction(self.action_next)
        self.menuFile.addAction(self.action_prev)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.image.setText(_translate("MainWindow", "(No image)"))
        self.target.setText(_translate("MainWindow", "(No text)"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.action_open_dataset.setText(_translate("MainWindow", "Open dataset"))
        self.action_open_dataset.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.action_next.setText(_translate("MainWindow", "Next"))
        self.action_next.setShortcut(_translate("MainWindow", "D"))
        self.action_prev.setText(_translate("MainWindow", "Prev"))
        self.action_prev.setShortcut(_translate("MainWindow", "A"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data", "-d")

    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    win = MainWindow(args.data)
    win.show()
    app.exec()
