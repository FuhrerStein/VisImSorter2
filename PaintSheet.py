from PyQt5 import QtGui
from PyQt5.QtCore import QRect, QPoint, Qt, QSize, QTimer
from PyQt5.QtGui import QPainter, QPen, QPixmap, QBrush, QPolygon, QRadialGradient, QColor
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
        self.active_frame = None
        self.hide_marks_timer = QTimer()
        self.hide_marks_timer.timeout.connect(self.hihe_marks)
        self.hide_marks_timer.setInterval(1800)
        self.hide_marks_timer.setSingleShot(True)
        self.force_sizes = None
        self.img_zoom = 1
        self.img_xy = QPoint(0, 0)
        self.pressed_mouse = None
        self.adjustment_mode = None
        self.mouse_press_point = None
        self.suggest_marks = None

    def hihe_marks(self):
        self.pos = None
        self.update()

    def paintEvent(self, event):
        self.painter = QPainter(self)
        separator_x = self.width() * self.separator_line
        # green_brush = QBrush(Qt.green)
        # red_brush = QBrush(Qt.red)
        # gray_brush = QBrush(Qt.gray)
        green_brush = QBrush(QColor(0, 255, 0, 60))
        red_brush = QBrush(QColor(255, 0, 0, 60))
        gray_brush = QBrush(QColor(120, 120, 120, 60))
        green_color = 0, 255, 0
        red_color = 255, 0, 0
        gray_color = 120, 120, 120
        # painter.setBrush(QBrush(QColor(*brush_color, brush_alpha)))

        black_dot_line_pen = QPen(Qt.black, 6, Qt.DotLine)
        black_solid_pen = QPen(Qt.black, 2, Qt.SolidLine)
        black_bold_pen = QPen(Qt.black, 6, Qt.SolidLine)

        if self.active_frame is not None and type(self.pixmap_r) is QPixmap:
            size = self.pixmap_r.size() * (self.width() / self.pixmap_r.width())
            start_y = (self.height() - size.height()) / 2 + self.img_xy.y() / self.height() * size.height()
            draw_rect = QRect(0, start_y, size.width(), size.height())
            self.painter.drawPixmap(draw_rect, self.pixmap_r.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            size = self.pixmap_r.size() * min((separator_x - 10) / self.pixmap_r.width(), self.height() / self.pixmap_r.height())
            start_x = self.width() - separator_x + 10
            start_point = QPoint(start_x, (self.height() - size.height()) / 2)
            draw_rect = QRect(start_point, size)
            self.painter.drawPixmap(draw_rect, self.pixmap_r.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if type(self.pixmap_l) is QPixmap:
            size = self.pixmap_l.size() * min((separator_x - 10) / self.pixmap_l.width(), self.height() / self.pixmap_l.height())
            start_point = QPoint(separator_x - size.width() - 10, (self.height() - size.height()) / 2)

            draw_rect = QRect(start_point, size)
            self.painter.drawPixmap(draw_rect, self.pixmap_l.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if type(self.pixmap_r) is QPixmap and self.active_frame is None:
            size = self.pixmap_r.size() * min(self.width() * (1 - self.separator_line) / self.pixmap_r.width(), self.height() / self.pixmap_r.height())
            # size *= self.img_zoom
            start_x = max(separator_x + 10, (self.width() - size.width()) / 2)
            start_point = QPoint(start_x, (self.height() - size.height()) / 2)

            draw_rect = QRect(start_point, size)
            self.painter.drawPixmap(draw_rect, self.pixmap_r.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def draw_dot(x, i, selected):
            color = [red_color, gray_color, green_color][self.suggest_marks[i] + 1]
            opacity = 80 if selected else 60
            self.painter.setBrush(QBrush(QColor(*color, opacity)))

            self.painter.setPen(black_solid_pen if selected else black_bold_pen)
            radius = 15 if selected else 25
            point_x = self.width() / 2 + x - radius
            self.painter.drawEllipse(point_x, 40 - radius, 2 * radius, 2 * radius)

        if self.active_frame is not None and self.suggest_marks:
            draw_dot(-80, 0, self.active_frame == 1)
            draw_dot(80, 1, self.active_frame == 0)

        if self.pos:
            self.painter.setBrush(QBrush(Qt.blue))
            self.painter.drawEllipse(self.pos, 5, 5)

        if self.adjustment_mode == 3:
            self.painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
            self.painter.drawLine(separator_x, 0, separator_x, self.height())

        self.painter.end()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        self.pos = event.pos()
        if self.pressed_mouse == 1:
            self.adjustment_mode = 1
        elif self.pressed_mouse == 2:
            self.adjustment_mode = 3

        if self.adjustment_mode == 1:
            point_diff = event.pos() - self.mouse_press_point
            point_diff *= point_diff.manhattanLength() ** .3
            self.img_xy += point_diff
            self.img_xy.setY(self.img_xy + self.size().height() // 2)
            print(self.img_xy)
        elif self.adjustment_mode == 2:
            new_zoom = self.img_zoom * ((self.mouse_press_point.y() - event.y()) / 200 + 1)
            zoom_coef = new_zoom / self.img_zoom - 1
            xy_correction = QPoint(self.width() * (self.separator_line - .5) + 10, 10)
            self.img_xy += (self.img_xy + xy_correction / 2) * zoom_coef
            self.img_zoom = new_zoom
        elif self.adjustment_mode == 3:
            self.separator_line -= (self.mouse_press_point.x() - event.x()) / self.width()

        if self.adjustment_mode:
            self.mouse_press_point = event.pos()

        self.hide_marks_timer.start()
        self.update()
        QApplication.processEvents()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.pressed_mouse = event.button()
        self.mouse_press_point = event.pos()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.pressed_mouse = None
        self.mouse_press_point = None
        self.adjustment_mode = None


class PreviewSheet(QWidget):
    def __init__(self, parent=None, switch_func=None, mark_func=None):
        QWidget.__init__(self, parent=parent)
        self.pixmaps = [None] * 15
        self.delete_marks = [None] * 15
        self.suggest_marks = [None] * 15
        self.central_pixmap = 0
        self.cursor_pos = None
        self.thumb_hover = None
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
        self.thumb_hover = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        pen_color = Qt.green
        pen_width, pen_line = 4, Qt.SolidLine
        green_dot_line_pen = QPen(Qt.green, 4, Qt.DotLine)
        red_dot_line_pen = QPen(Qt.red, 4, Qt.DotLine)
        black_dot_line_pen = QPen(Qt.black, 4, Qt.DotLine)
        green_solid_pen = QPen(Qt.green, 2, Qt.SolidLine)
        green_fat_solid_pen = QPen(Qt.green, 4, Qt.SolidLine)
        red_solid_pen = QPen(Qt.red, 2, Qt.SolidLine)
        black_solid_pen = QPen(Qt.black, 2, Qt.SolidLine)
        white_solid_pen = QPen(Qt.white, 4, Qt.SolidLine)

        # green_grad = QRadialGradient(1, 3, 8)
        # green_grad.setColorAt(0, Qt.green)
        # green_grad.setColorAt(15, Qt.cyan)
        # green_solid_brush = QBrush(Qt.green)
        # red_solid_brush = QBrush(Qt.red)
        # green_solid_brush = QBrush(Qt.green)
        # red_solid_brush = QBrush(Qt.red)
        no_brush = QBrush(Qt.NoBrush)
        last_pen = None
        last_pen_style = (None, None, None)

        def set_this_pen(pen, force=False):
            nonlocal last_pen
            if last_pen != pen or force:
                painter.setPen(pen)
                last_pen = pen

        def set_this_penstyle(pen_color, pen_width, pen_line, force=False):
            nonlocal last_pen
            if (pen_color, pen_width, pen_line) != last_pen_style or force:
                painter.setPen(QPen(pen_color, pen_width, pen_line))
                last_pen = (pen_color, pen_width, pen_line)

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
                if position == self.thumb_hover:
                    set_this_pen(white_solid_pen)
                    painter.drawRect(draw_rect)
                    set_this_pen(black_dot_line_pen)
                else:
                    set_this_pen(black_solid_pen)
            painter.drawRect(draw_rect)

            def draw_mark(mark, mark_center):
                if mark:
                    brush_color = (255 * (mark == -1), 255 * (mark == 1), 0)
                    brush_alpha = 80 if self.delete_marks[idx] else 255
                    if position > 0:
                        brush_alpha = 80
                    painter.setBrush(QBrush(QColor(*brush_color, brush_alpha)))
                    painter.drawEllipse(mark_center, 10, 10)

            if self.suggest_marks[idx]:
                left_mark, right_mark = self.suggest_marks[idx]
                left_mark_center = thumb_center + QPoint(-50, -50)
                right_mark_center = thumb_center + QPoint(50, -50)
                set_this_pen(black_solid_pen)
                draw_mark(left_mark, left_mark_center)
                draw_mark(right_mark, right_mark_center)
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
        self.thumb_hover = thumb_id
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


