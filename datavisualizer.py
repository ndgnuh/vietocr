import re
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib
import sys
from dataclasses import dataclass, field
from typing import *
from functools import *
from PyQt5.QtCore import (
    QSize,
)
from PyQt5.QtGui import (QPalette, QColor)
from PyQt5.QtWidgets import (
    QApplication,
    QPushButton,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QAction,
    QFileDialog,
    QWidget,
    QLabel,
    QGridLayout
)
from PyQt5.QtGui import QKeySequence
from PyQt5 import QtWidgets as qw
from os import path
WINDOW_TITLE = "Data visualizer"


def split_annotation(line):
    splits = re.split(r"\s+", line.strip())
    return splits[0], " ".join(splits[1:])


def meaningful_path(file):
    thisdir = path.dirname(__file__)
    common = path.commonpath((file, thisdir))
    return file.replace(common, "").lstrip("/")


matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.cla()
        super().__init__(self.fig)

    def show_data(self, image, label):
        axes = self.axes
        axes.cla()
        axes.imshow(image)
        axes.set_title(label)
        self.figure.canvas.draw()
        # self.figure = fig


@dataclass
class State:
    annotation_file: Optional[str] = None
    index: Optional[int] = None
    annotations: List[str] = field(default_factory=list)
    total: int = -1

    @property
    def data_root(self):
        return path.dirname(self.annotation_file)

    def set_index(self, index):
        index = max(index, 0)
        index = min(index, self.total - 1)
        self.index = index

    def set_annotations(self, file):
        self.annotation_file = file
        with open(file) as f:
            self.annotations = [
                split_annotation(line)
                for line in f.readlines()
            ]
        self.total = len(self.annotations)
        self.index = 0


def vbox(widgets):
    layout = QVBoxLayout()
    for w in widgets:
        if isinstance(w, (QVBoxLayout, QHBoxLayout)):
            layout.addLayout(w)
        else:
            layout.addWidget(w)
    return layout


def hbox(widgets):
    layout = QHBoxLayout()
    for w in widgets:
        if isinstance(w, (QVBoxLayout, QHBoxLayout)):
            layout.addLayout(w)
        else:
            layout.addWidget(w)
    return layout


class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(WINDOW_TITLE)
        self.setFixedSize(QSize(720, 480))
        self.setup_toolbar()
        self.state = State()

        # Widgets
        self.info = QLabel("")
        btn_next = QPushButton(">")
        btn_next.clicked.connect(self.next)
        btn_prev = QPushButton("<")
        btn_prev.clicked.connect(self.prev)
        btn_aug = QPushButton("Augment")
        btn_refresh = QPushButton("Refresh")
        canvas = MplCanvas(self)

        # Layout
        root_layout = vbox([
            self.info,
            canvas,
            hbox([
                btn_prev, btn_next, btn_aug, btn_refresh
            ]),
        ])

        # Shortcuts
        shortcuts = [
            ("Ctrl+o", self.load_data),
            ("d", self.next),
            ("a", self.prev),
            ("Shift+a", partial(self.next, -10)),
            ("Shift+d", partial(self.next, 10)),
        ]
        for k, h in shortcuts:
            ks = qw.QShortcut(QKeySequence(k), self)
            ks.activated.connect(h)

        # Setup
        self.canvas = canvas
        root = QWidget(self)
        root.setLayout(root_layout)
        self.setCentralWidget(root)

    def setup_toolbar(self):
        toolbar = QToolBar()

        # actions
        load_data = QAction("Load data", self)
        load_data.triggered.connect(self.load_data)

        # setup
        self.addToolBar(toolbar)
        toolbar.addAction(load_data)

    def load_data(self, *args):
        picker = QFileDialog(self)
        file, _ = picker.getOpenFileName(
            self,
            filter="*.txt",
            caption="Select annotation file",
            directory="data/"
        )
        if file is None:
            return

        self.state.set_annotations(file)
        self.update()

    def next(self, offset=1):
        self.state.set_index(self.state.index + offset)
        self.update()

    def prev(self):
        self.state.set_index(self.state.index - 1)
        self.update()

    def update(self):
        file = self.state.annotation_file
        anns = self.state.annotations
        print(file)

        if file is None:
            return

        idx = self.state.index
        data_root = self.state.data_root
        image_file = path.join(data_root, anns[idx][0])
        label = anns[idx][1]
        total = self.state.total
        self.info.setText(
            f"Annotation: {meaningful_path(file)} [{idx + 1}/{total}]"
        )
        print(image_file)
        self.canvas.show_data(
            Image.open(image_file),
            label
        )
        # self.plt.imshow(Image.open(image_file))
        # self.plt.set_title(label)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
