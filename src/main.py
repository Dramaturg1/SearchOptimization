from PySide6.QtWidgets import QApplication
from src.core.CustomLoader import CustomLoader
import sys

app = QApplication(sys.argv)

loader = CustomLoader()
window = loader.load("src/ui/main.ui")

window.show()
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]
window.widget.plot(x, y)

sys.exit(app.exec())