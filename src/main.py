from PySide6.QtWidgets import QApplication
from src.core.CustomLoader import CustomLoader
import sys
import os

app = QApplication(sys.argv)

loader = CustomLoader()
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_path = os.path.join(current_dir, "ui", "main.ui")
window = loader.load(ui_path)

window.show()
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]
window.widget.plot(x, y)

sys.exit(app.exec())