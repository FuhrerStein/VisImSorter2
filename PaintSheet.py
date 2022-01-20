from PyQt5 import QtGui
from PyQt5.QtCore import QRect, QPoint, Qt, QSize, QTimer
from PyQt5.QtGui import QPainter, QPen, QPixmap, QBrush, QPolygon, QRadialGradient
from PyQt5.QtWidgets import QWidget, QApplication


class PaintSheet(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.pixmap_l = None
        self.pixmap_r = None
        self.pos = None
        self.setMouseTracking(True)
        self.painter = QPainter()
        self.separate_pictures = False
        self.separator_line = .5
        self.separator_mode = False
        self.active_frame = None
        self.hide_marks_timer = QTimer()
        self.hide_marks_timer.timeout.connect(self.hihe_marks)
        self.hide_marks_timer.setInterval(1800)
        self.hide_marks_timer.setSingleShot(True)
        self.force_sizes = None

    def hihe_marks(self):
        self.pos = None
        self.update()

    def paintEvent(self, event):
        self.painter = QPainter(self)
        if self.force_sizes:
            size_l = self.force_sizes[0] * min(self.width() * (1 - self.separator_line) / self.force_sizes[0].width(),
                                              self.height() / self.force_sizes[0].height())
            size_r = self.force_sizes[1] * min(self.width() * (1 - self.separator_line) / self.force_sizes[1].width(),
                                              self.height() / self.force_sizes[1].height())
            width_forced = max(size_l.width(), size_r.width())

        if type(self.pixmap_l) is QPixmap:
            size = self.pixmap_l.size() * min(self.width() * self.separator_line / self.pixmap_l.width(), self.height() / self.pixmap_l.height())
            start_point = QPoint(self.width() * self.separator_line - size.width() - 10, (self.height() - size.height()) / 2)

            draw_rect = QRect(start_point, size)
            self.painter.drawPixmap(draw_rect, self.pixmap_l.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if type(self.pixmap_r) is QPixmap:
            size = self.pixmap_r.size() * min(self.width() * (1 - self.separator_line) / self.pixmap_r.width(), self.height() / self.pixmap_r.height())
            start_point = QPoint(self.width() * self.separator_line + 10, (self.height() - size.height()) / 2)
            if self.force_sizes:
                start_point.setX(self.width() * self.separator_line + 10 + (width_forced - size.width()) / 2)
            draw_rect = QRect(start_point, size)
            self.painter.drawPixmap(draw_rect, self.pixmap_r.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if self.active_frame is not None:
                green_solid_brush = QBrush(Qt.green)
                self.painter.setBrush(green_solid_brush)
                if self.active_frame == 0:
                    point_x = self.width() * self.separator_line + 20
                else:
                    point_x = start_point.x() + size.width() - 40
                self.painter.drawEllipse(point_x + 2, 10, 20, 20)
                black_solid_pen = QPen(Qt.black, 2, Qt.SolidLine)
                self.painter.setPen(black_solid_pen)
                self.painter.drawEllipse(point_x, 8, 24, 24)

        if self.pos:
            self.painter.setBrush(QBrush(Qt.blue))
            self.painter.drawEllipse(self.pos, 5, 5)

        if self.separator_mode:
            self.painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
            x_pos = self.width() * self.separator_line
            self.painter.drawLine(x_pos, 0, x_pos, self.height())

        self.painter.end()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        self.pos = event.pos()
        if self.separator_mode:
            self.separator_line = event.x() / self.width()
        self.hide_marks_timer.start()
        QApplication.processEvents()
        self.update()
        QApplication.processEvents()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == 2:
            self.separator_mode = True

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.button() == 2:
            self.separator_mode = False


class PreviewSheet(QWidget):
    def __init__(self, parent=None, switch_func=None, mark_func=None):
        QWidget.__init__(self, parent=parent)
        self.pixmaps = [None] * 15
        self.delete_marks = [None] * 15
        self.suggest_marks = [None] * 15
        self.central_pixmap = 0
        self.cursor_pos = None
        self.setMouseTracking(True)
        self.painter = QPainter()
        self.switch = switch_func
        self.mark = mark_func
        self.mark_intent_l = 0
        self.mark_intent_r = 0
        self.hide_marks_timer = QTimer()
        self.hide_marks_timer.timeout.connect(self.hide_marks)
        self.hide_marks_timer.setInterval(1800)
        self.hide_marks_timer.setSingleShot(True)

    def hide_marks(self):
        self.cursor_pos = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        green_dot_line_pen = QPen(Qt.green, 4, Qt.DotLine)
        red_dot_line_pen = QPen(Qt.red, 4, Qt.DotLine)
        black_dot_line_pen = QPen(Qt.black, 4, Qt.DotLine)

        green_solid_pen = QPen(Qt.green, 2, Qt.SolidLine)
        green_fat_solid_pen = QPen(Qt.green, 4, Qt.SolidLine)
        red_solid_pen = QPen(Qt.red, 2, Qt.SolidLine)
        black_solid_pen = QPen(Qt.black, 2, Qt.SolidLine)

        # green_grad = QRadialGradient(1, 3, 8)
        # green_grad.setColorAt(0, Qt.green)
        # green_grad.setColorAt(15, Qt.cyan)
        green_solid_brush = QBrush(Qt.green)
        red_solid_brush = QBrush(Qt.red)
        no_brush = QBrush(Qt.NoBrush)
        last_pen = None

        def set_this_pen(pen, force=False):
            nonlocal last_pen
            if last_pen != pen or force:
                painter.setPen(pen)
                last_pen = pen

        def draw_thumb(position):
            idx = (self.central_pixmap + position) % 15
            thumb_center = QPoint(self.size().width() / 2 + position * 155, self.size().height() / 2)
            draw_rect = QRect(thumb_center.x() - 77, 2, 150, 150)
            set_this_pen(black_solid_pen if position else green_dot_line_pen)

            if thumb_center.x() < -80 or thumb_center.x() > 80 + self.size().width():
                return

            if type(self.pixmaps[idx]) is QPixmap:
                painter.drawPixmap(draw_rect, self.pixmaps[idx])

            if self.delete_marks[idx]:
                left_mark = self.delete_marks[idx][0]
                right_mark = self.delete_marks[idx][1]
                left_mark_center = thumb_center + QPoint(-50, 50)
                right_mark_center = thumb_center + QPoint(50, 50)
                if left_mark == 1:
                    set_this_pen(green_solid_pen)
                    painter.drawLines((left_mark_center - QPoint(10, 0)), (left_mark_center + QPoint(10, 0)),
                                      (left_mark_center - QPoint(0, 10)), (left_mark_center + QPoint(0, 10)))
                elif left_mark == -1:
                    set_this_pen(red_solid_pen)
                    painter.drawLines((left_mark_center - QPoint(10, 0)), (left_mark_center + QPoint(10, 0)))

                elif left_mark == -2:
                    set_this_pen(red_solid_pen)
                    painter.drawLines((left_mark_center - QPoint(-10, 30)), (left_mark_center - QPoint(10, 30)))

                if right_mark == 1:
                    set_this_pen(green_solid_pen)
                    painter.drawLines((right_mark_center - QPoint(10, 0)), (right_mark_center + QPoint(10, 0)),
                                      (right_mark_center - QPoint(0, 10)), (right_mark_center + QPoint(0, 10)))
                elif right_mark == -1:
                    set_this_pen(red_solid_pen)
                    painter.drawLines((right_mark_center - QPoint(10, 0)), (right_mark_center + QPoint(10, 0)))

                elif right_mark == -2:
                    set_this_pen(red_solid_pen)
                    painter.drawLines((right_mark_center - QPoint(-10, 30)), (right_mark_center - QPoint(10, 30)))

                if position:
                    set_this_pen(red_solid_pen)
                else:
                    set_this_pen(red_dot_line_pen)
            else:
                if position:
                    set_this_pen(black_solid_pen)
                else:
                    set_this_pen(black_dot_line_pen)
            painter.drawRect(draw_rect)

            if self.suggest_marks[idx]:
                left_mark, right_mark = self.suggest_marks[idx]
                left_mark_center = thumb_center + QPoint(-50, -50)
                right_mark_center = thumb_center + QPoint(50, -50)
                if left_mark:
                    painter.setBrush(green_solid_brush if left_mark == 1 else red_solid_brush)
                    set_this_pen(black_solid_pen)
                    painter.drawEllipse(left_mark_center, 10, 10)
                if right_mark:
                    painter.setBrush(green_solid_brush if right_mark == 1 else red_solid_brush)
                    set_this_pen(black_solid_pen)
                    painter.drawEllipse(right_mark_center, 10, 10)
                painter.setBrush(no_brush)

            if not position:
                set_this_pen(green_fat_solid_pen)
                ctr_x = self.size().width() / 2
                triangle = QPolygon()
                triangle << QPoint(ctr_x, 15) << QPoint(ctr_x - 5, 2) << QPoint(ctr_x + 5, 2) # << QPoint(ctr_x, 15)
                painter.drawPolygon(triangle)

        for i in range(1, 16):
            thumb_id = i // 2 if i % 2 else -i // 2
            draw_thumb(thumb_id)

        if self.cursor_pos:
            left_center = self.cursor_pos - QPoint(15, 0)
            right_center = self.cursor_pos + QPoint(15, 0)
            if self.mark_intent_l == 0:
                painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
                painter.drawEllipse(left_center, 10, 10)
            elif self.mark_intent_l == 1:
                painter.setPen(green_solid_pen)
                painter.drawLines((left_center - QPoint(10, 0)), (left_center + QPoint(10, 0)), (left_center - QPoint(0, 10)), (left_center + QPoint(0, 10)))
            elif self.mark_intent_l == -1:
                painter.setPen(red_solid_pen)
                painter.drawLines((left_center - QPoint(10, 0)), (left_center + QPoint(10, 0)))

            if self.mark_intent_r == 0:
                painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
                painter.drawEllipse(right_center, 10, 10)
            elif self.mark_intent_r == 1:
                painter.setPen(green_solid_pen)
                painter.drawLines((right_center - QPoint(10, 0)), (right_center + QPoint(10, 0)), (right_center - QPoint(0, 10)), (right_center + QPoint(0, 10)))
            elif self.mark_intent_r == -1:
                painter.setPen(red_solid_pen)
                painter.drawLines((right_center - QPoint(10, 0)), (right_center + QPoint(10, 0)))

    def mouseMoveEvent(self, event):
        thumb_id = round((event.pos().x() - self.size().width() / 2) / 155)
        thumb_center_x = self.size().width() / 2 + thumb_id * 155
        thumb_center_y = self.size().height() / 2

        off_center_x = event.pos().x() - thumb_center_x
        off_center_y = event.pos().y() - thumb_center_y
        self.mark_intent_l = 1 if off_center_x < -25 else -1
        self.mark_intent_r = 1 if off_center_x > 25 else -1
        if abs(off_center_x) < 25:
            if abs(off_center_y) < 25:
                self.mark_intent_l = 0
                self.mark_intent_r = 0
            elif off_center_y < 0:
                self.mark_intent_l = 1
                self.mark_intent_r = 1
            elif off_center_y > 0:
                self.mark_intent_l = -1
                self.mark_intent_r = -1
        self.cursor_pos = event.pos()
        self.hide_marks_timer.start()
        QApplication.processEvents()
        self.update()
        QApplication.processEvents()

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        thumb = int((a0.x() - self.size().width() / 2 + 75) // 155)
        if a0.button() == Qt.RightButton:
            self.mark(thumb, self.mark_intent_l, self.mark_intent_r)
        elif a0.button() == Qt.LeftButton:
            self.switch(thumb)


