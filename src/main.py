from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import QUiLoader
import sys

app = QApplication(sys.argv)

loader = QUiLoader()
window = loader.load("src/ui/main.ui")

window.show()

sys.exit(app.exec())