"""PySide6 desktop interface: upload, manual ROI, classify, chart, export."""
import sys
from pathlib import Path
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QRectF, Signal, Slot, QObject, QThread
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QProgressBar, QAbstractItemView)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from inference import Predictor, export_results

ROOT = Path(__file__).resolve().parents[1]


class ImagePanel(QWidget):
    selection_changed = Signal()

    def __init__(self):
        super().__init__()
        self.image = self.qimage = self.box = self.start = None
        self.setMinimumSize(360, 280)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image):
        self.image = image
        rgb = image.convert('RGB')
        self.qimage = QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()
        self.box = None
        self.update()

    def image_rect(self):
        if self.image is None:
            return QRectF()
        scale = min((self.width()-32)/self.image.width, (self.height()-32)/self.image.height)
        w, h = self.image.width*scale, self.image.height*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def point(self, position):
        rect = self.image_rect()
        x = max(0, min(self.image.width, (position.x()-rect.x())/rect.width()*self.image.width))
        y = max(0, min(self.image.height, (position.y()-rect.y())/rect.height()*self.image.height))
        return round(x), round(y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#edf2f7'))
        if self.qimage is None:
            painter.setPen(QColor('#53657c'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '사진을 업로드하세요\nPNG · JPG · BMP · WEBP · PPM')
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.qimage)
        if self.box:
            x1, y1, x2, y2 = self.box
            sx, sy = rect.width()/self.image.width, rect.height()/self.image.height
            painter.setPen(QPen(QColor('#007e87'), 3))
            painter.drawRect(QRectF(rect.x()+x1*sx, rect.y()+y1*sy, (x2-x1)*sx, (y2-y1)*sy))

    def mousePressEvent(self, event):
        if self.image is not None and event.button() == Qt.MouseButton.LeftButton and self.image_rect().contains(event.position()):
            self.start = self.point(event.position())

    def mouseMoveEvent(self, event):
        if self.start:
            x, y = self.point(event.position())
            a, b = self.start
            self.box = (min(a,x), min(b,y), max(a,x), max(b,y))
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start:
            self.mouseMoveEvent(event)
            self.start = None
            if self.box[2]-self.box[0] < 3 or self.box[3]-self.box[1] < 3:
                self.box = None
            self.selection_changed.emit()
            self.update()

    def reset(self):
        self.box = None
        self.selection_changed.emit()
        self.update()


class InferenceWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.predictor = Predictor()

    @Slot(object)
    def run(self, request):
        try:
            self.completed.emit(self.predictor.run(*request))
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Traffic Sign Studio | GTSRB')
        self.resize(1200, 850)
        self.image = self.image_path = self.result = None
        self.busy = False
        self.model_path = ROOT / 'runs/cnn/best.keras'
        self.thread = QThread(self)
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.thread)
        self.requested.connect(self.worker.run)
        self.worker.completed.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        title = QLabel('Traffic Sign Studio')
        title.setStyleSheet('font-size:28px; font-weight:700; color:#122940')
        layout.addWidget(title)
        layout.addWidget(QLabel('사진 업로드 → 표지판 영역 선택 → 43종 분류 → 결과 저장'))
        toolbar = QHBoxLayout()
        self.upload = QPushButton('사진 업로드')
        self.choose = QPushButton('모델 선택')
        self.clear_roi = QPushButton('전체 이미지 사용')
        self.analyze = QPushButton('표지판 인식')
        self.analyze.setStyleSheet('background:#007e87;color:white;font-weight:700')
        self.save = QPushButton('결과 CSV 저장')
        self.save_plot = QPushButton('그래프 PNG 저장')
        for button in (self.upload, self.choose, self.clear_roi, self.analyze, self.save, self.save_plot):
            toolbar.addWidget(button)
        layout.addLayout(toolbar)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        self.update_model_label()
        layout.addWidget(self.model_label)
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.panel = ImagePanel()
        left_layout.addWidget(self.panel, 1)
        self.image_label = QLabel('선택한 사진이 없습니다.')
        self.image_label.setWordWrap(True)
        left_layout.addWidget(self.image_label)
        self.roi_label = QLabel('전체 이미지 • 드래그하면 표지판 영역을 직접 선택합니다.')
        self.roi_label.setWordWrap(True)
        left_layout.addWidget(self.roi_label)
        split.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.prediction = QLabel('사진을 선택하고 인식을 시작하세요.')
        self.prediction.setWordWrap(True)
        self.prediction.setStyleSheet('font-size:19px; font-weight:600; padding:12px; background:#e8f5f4')
        right_layout.addWidget(self.prediction)
        self.figure = Figure(figsize=(5, 3), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(230)
        right_layout.addWidget(self.canvas)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['클래스 ID', '표지판 이름', '점수'])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.table, 1)
        split.addWidget(right)
        split.setSizes([520, 620])
        layout.addWidget(split, 1)
        note = QLabel('현재 기능: 선택 영역 분류. 자동 위치 검출은 지원하지 않습니다. 점수는 보정된 정답 확률이 아닙니다.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(6)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.upload.clicked.connect(self.open_image)
        self.choose.clicked.connect(self.open_model)
        self.clear_roi.clicked.connect(self.panel.reset)
        self.panel.selection_changed.connect(self.invalidate)
        self.analyze.clicked.connect(self.start_inference)
        self.save.clicked.connect(self.save_csv)
        self.save_plot.clicked.connect(self.save_chart)
        self.invalidate()
        self.setStyleSheet('QMainWindow{background:#f8fafc} QLabel{color:#23384e} QPushButton{padding:10px;border:1px solid #cbd5e1;border-radius:6px;background:white} QPushButton:disabled{color:#9aa6b2;background:#eef2f6} QTableWidget{background:white;border:1px solid #d9e2ec}')

    def update_model_label(self):
        self.model_label.setText(f'모델: {self.model_path.name}  |  {"준비됨" if self.model_path.is_file() else "파일 없음 — 모델을 선택하세요"}')
        self.model_label.setToolTip(str(self.model_path))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '표지판 사진 선택', '', 'Images (*.png *.jpg *.jpeg *.bmp *.webp *.ppm)')
        if path:
            try:
                self.load_image(path)
            except Exception as error:
                QMessageBox.warning(self, '이미지 오류', str(error))

    def load_image(self, path):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        self.image, self.image_path = image, Path(path)
        self.panel.set_image(image)
        self.image_label.setText(f'{self.image_path.name} · {image.width} × {image.height} px')
        self.invalidate()

    def open_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'GTSRB 모델 선택', str(ROOT / 'runs'), 'Keras model (*.keras)')
        if path:
            self.model_path = Path(path)
            self.update_model_label()
            self.invalidate()

    def set_busy(self, value):
        self.busy = value
        for widget in (self.upload, self.choose, self.clear_roi, self.panel):
            widget.setEnabled(not value)
        self.analyze.setEnabled(not value and self.image is not None and self.model_path.is_file())
        self.save.setEnabled(not value and self.result is not None)
        self.save_plot.setEnabled(not value and self.result is not None)
        self.progress.setRange(0, 0 if value else 1)
        self.progress.setValue(0)

    def invalidate(self):
        self.result = None
        self.table.setRowCount(0)
        self.figure.clear()
        self.canvas.draw_idle()
        self.prediction.setText('선택 영역을 확인하고 표지판 인식을 누르세요.')
        self.roi_label.setText(f'직접 선택한 영역 (x1, y1, x2, y2): {self.panel.box}' if self.panel.box else '전체 이미지 • 드래그하면 표지판 영역을 직접 선택합니다.')
        self.set_busy(False)

    def start_inference(self):
        if self.busy or self.image is None:
            return
        self.result = None
        self.set_busy(True)
        self.prediction.setText('분석 중… 최초 실행은 모델 로딩으로 시간이 걸릴 수 있습니다.')
        self.requested.emit((str(self.model_path.resolve()), self.image.copy(), self.panel.box))

    @Slot(object)
    def show_result(self, frame):
        self.result = frame
        best = frame.iloc[0]
        self.prediction.setText(f"{best['class_name']}\nClass {int(best['class_id'])} · 점수 {best['score']:.2%}")
        self.table.setRowCount(len(frame))
        for i, row in frame.iterrows():
            for j, value in enumerate((str(row['class_id']), row['class_name'], f"{row['score']:.2%}")):
                self.table.setItem(i, j, QTableWidgetItem(value))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        top = frame.head(5).iloc[::-1]
        ax.barh([f"Class {c}" for c in top.class_id], top.score*100, color='#007e87')
        ax.set(xlim=(0, 100), xlabel='Softmax score (%)', title='Top 5 predictions')
        ax.spines[['top', 'right']].set_visible(False)
        self.canvas.draw_idle()
        self.set_busy(False)
        self.statusBar().showMessage('분석 완료. 전체 43개 클래스 결과를 CSV로 저장할 수 있습니다.')

    @Slot(str)
    def show_error(self, message):
        self.invalidate()
        self.prediction.setText('분석에 실패했습니다. 사진과 모델을 확인하세요.')
        QMessageBox.warning(self, '분석 오류', message)

    def save_csv(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, '결과 저장', 'prediction.csv', 'CSV (*.csv)')
        if path:
            try:
                export_results(self.result, path, self.image_path, self.model_path, self.panel.box)
                self.statusBar().showMessage(f'CSV 저장 완료: {path}')
            except Exception as error:
                QMessageBox.warning(self, '저장 오류', str(error))

    def save_chart(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, '그래프 저장', 'top5.png', 'PNG (*.png)')
        if path:
            try:
                self.figure.savefig(path, dpi=160)
            except Exception as error:
                QMessageBox.warning(self, '저장 오류', str(error))

    def closeEvent(self, event):
        if self.busy:
            self.statusBar().showMessage('분석이 끝난 뒤 창을 닫아주세요.')
            event.ignore()
            return
        self.thread.quit()
        self.thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
