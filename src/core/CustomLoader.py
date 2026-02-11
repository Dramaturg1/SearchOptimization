from PySide6.QtUiTools import QUiLoader
import pyqtgraph as pg

class CustomLoader(QUiLoader):
    def createWidget(self, classname, parent=None, name=''):
        if classname == 'PlotWidget':
            return pg.PlotWidget(parent)
        return super().createWidget(classname, parent, name)