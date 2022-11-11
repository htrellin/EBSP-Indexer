from os import path
from kikuchipy import load, filters, generators
from PySide6.QtWidgets import QDialog

from utils.filebrowser import FileBrowser
from utils.worker import Worker
from ui.ui_pattern_processing_dialog import Ui_PatternProcessingWindow


class PatternProcessingDialog(QDialog):
    def __init__(self, parent=None, pattern_path=None):
        super().__init__(parent)

        self.threadPool = parent.threadPool
        self.working_dir = path.dirname(pattern_path)

        self.pattern_path = pattern_path

        self.filenamebase = path.basename(self.pattern_path).split(".")[0]

        # Standard filename of processed pattern
        self.save_path = path.join(
            self.working_dir, f"{self.filenamebase}_processed.h5"
        )

        self.ui = Ui_PatternProcessingWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(f"{self.windowTitle()} - {self.pattern_path}")
        self.setupConnections()

        try:
            self.s = load(self.pattern_path, lazy=True)
        except Exception as e:
            raise e

        self.showImage(self.s.inav[1, 1])

        self.gaussian_window = filters.Window("gaussian", std=1)

        self.options = self.getOptions()

        self.fileBrowser = FileBrowser(
            mode=FileBrowser.SaveFile,
            dirpath=self.working_dir,
            filter_name="Hierarchical Data Format (*.h5);;NordifUF Pattern Files (*.dat)",
        )

    def setupConnections(self):
        self.ui.browseButton.clicked.connect(lambda: self.setSavePath())
        self.ui.buttonBox.accepted.connect(lambda: self.run_processing())
        self.ui.buttonBox.rejected.connect(lambda: self.reject())
        self.ui.folderEdit.setText(self.working_dir)
        self.ui.filenameEdit.setText(path.basename(self.save_path))

        # Whenever user checks/unchecks boxes the preview window updates to show the result of current choices
        self.ui.staticBackgroundBox.stateChanged.connect(
            lambda: self.preview_processing()
        )
        self.ui.dynamicBackgroundBox.stateChanged.connect(
            lambda: self.preview_processing()
        )
        self.ui.averageBox.stateChanged.connect(lambda: self.preview_processing())
        

    def setSavePath(self):
        if self.fileBrowser.getFile():
            self.save_path = self.fileBrowser.getPaths()[0]
            self.ui.folderEdit.setText(path.dirname(self.save_path))
            self.ui.filenameEdit.setText(path.basename(self.save_path))

    def getOptions(self) -> dict:
        return {
            "static": {
                self.ui.staticBackgroundBox.isChecked(),
                self.s.remove_static_background(),
            },
            "dynamic": {
                self.ui.dynamicBackgroundBox.isChecked(),
                self.s.remove_dynamic_background(),
            },
            "average": {
                self.ui.averageBox.isChecked(),
                self.s.average_neighbour_patterns(self.gaussian_window),
            },
        }

    def remove_static(self, dataset):
        dataset.remove_static_background(show_progressbar=True)

    def remove_dynamic(self, dataset):
        dataset.remove_dynamic_background(show_progressbar=True)

    def average_neighbour(self, dataset):
        dataset.average_neighbour_patterns(self.gaussian_window, show_progressbar=True)

    def showImage(self, dataset):

        self.ui.previewWidget.vbl.setContentsMargins(0, 0, 0, 0)
        self.ui.previewWidget.canvas.ax.clear()
        self.ui.previewWidget.canvas.ax.axis(False)
        self.ui.previewWidget.canvas.ax.imshow(dataset.data, cmap="gray")
        self.ui.previewWidget.canvas.draw()

    def preview_processing(self):

        self.s_prev = self.s.inav[0:3, 0:3]
        self.options = self.getOptions()

        if self.ui.staticBackgroundBox.isChecked():
            self.remove_static(dataset=self.s_prev)
        if self.ui.dynamicBackgroundBox.isChecked():
            self.remove_dynamic(dataset=self.s_prev)
        if self.ui.averageBox.isChecked():
            self.average_neighbour(dataset=self.s_prev)

        self.showImage(self.s_prev.inav[1, 1])

        del self.s_prev

    def run_processing(self):
        # Pass the function to execute
        worker = Worker(self.apply_processing)
        # Execute
        self.threadPool.start(worker)
        self.accept()

    def apply_processing(self):

        print("Applying processing ...")
        self.options = self.getOptions()

        if self.options["static"]:
            print(f"static : {self.options['static']}")
            self.remove_static(dataset=self.s)
        if self.options["dynamic"]:
            print(f"dynamic : {self.options['dynamic']}")
            self.remove_dynamic(dataset=self.s)
        if self.options["average"]:
            print(f"average : {self.options['average']}")
            self.average_neighbour(dataset=self.s)

        try:
            self.s.save(
                self.save_path,
                overwrite=True,
            )
            print("Processing complete")
        except Exception as e:
            print(f"Could not save processed pattern: {e}")
            self.reject()
