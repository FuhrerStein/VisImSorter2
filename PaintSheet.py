import typing
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QRect, QPoint, Qt, QSize, QTimer
from PyQt5.QtGui import QPainter, QPen, QPixmap, QBrush, QPolygon, QRadialGradient, QColor
from PyQt5.QtWidgets import QWidget, QApplication, QSizePolicy
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsRectItem, QVBoxLayout, QGraphicsItem, QGraphicsEllipseItem, QHBoxLayout
from timeit import default_timer as timer


def restrict(val, min_val, max_val):
    if val < min_val: return min_val
    if val > max_val: return max_val
    return val


def mix(a, b, amount=.5):
    return a * (1 - amount) + b * amount


def smootherstep_ease(x):
    #  average between smoothstep and smootherstep
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    else:
        return x * x * (x * (x * (x * 6 - 15) + 8) + 3) / 2


def pixmap_from_image(im):
    qimage_obj = QtGui.QImage(im.tobytes(), im.width, im.height, im.width * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap(qimage_obj)


thumb_queue = {}


def receive_loaded_thumb(data):
    thumb_queue[data[0]] = data[1]


class PaintSheet(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.pixmap_l = None
        self.pixmap_r = None
        self.pos = None
        self.setMouseTracking(True)
        self.painter = QPainter()
        # self.separate_pictures = False
        self.separator_line = .5
        self.active_frame = None
        self.hide_marks_timer = QTimer()
        self.hide_marks_timer.timeout.connect(self.hihe_marks)
        self.hide_marks_timer.setInterval(1800)
        self.hide_marks_timer.setSingleShot(True)
        # self.force_sizes = None
        self.img_zoom = 1
        self.img_xy = QPoint(0, 0)
        self.pressed_mouse = None
        self.adjustment_mode = None
        self.mouse_press_point = None
        self.suggest_marks = None
        self.thumb_sheet = ThumbSheet_Scene

    def hihe_marks(self):
        self.pos = None
        self.update()

    def paintEvent(self, event):
        self.painter = QPainter(self)
        separator_x = int(self.width() * self.separator_line)
        # green_brush = QBrush(QColor(0, 255, 0, 60))
        # red_brush = QBrush(QColor(255, 0, 0, 60))
        # gray_brush = QBrush(QColor(120, 120, 120, 60))
        green_color = 0, 255, 0
        red_color = 255, 0, 0
        gray_color = 120, 120, 120

        black_dot_line_pen = QPen(Qt.black, 6, Qt.DotLine)
        black_solid_pen = QPen(Qt.black, 2, Qt.SolidLine)
        black_bold_pen = QPen(Qt.black, 6, Qt.SolidLine)

        if self.active_frame is not None and type(self.pixmap_r) is QPixmap:
            size = self.pixmap_r.size() * (self.width() / self.pixmap_r.width())
            start_y = int((self.height() - size.height()) / 2 + self.img_xy.y() / self.height() * size.height())
            draw_rect = QRect(0, start_y, int(size.width()), int(size.height()))
            self.painter.drawPixmap(draw_rect, self.pixmap_r.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            size = self.pixmap_r.size() * min((separator_x - 10) / self.pixmap_r.width(), self.height() / self.pixmap_r.height())
            start_x = self.width() - separator_x + 10
            start_point = QPoint(start_x, (self.height() - size.height()) // 2)
            draw_rect = QRect(start_point, size)
            self.painter.drawPixmap(draw_rect, self.pixmap_r.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if type(self.pixmap_l) is QPixmap:
            size = self.pixmap_l.size() * min((separator_x - 10) / self.pixmap_l.width(), self.height() / self.pixmap_l.height())
            start_point = QPoint(int(separator_x - size.width() - 10), (self.height() - size.height()) // 2)

            draw_rect = QRect(start_point, size)
            self.painter.drawPixmap(draw_rect, self.pixmap_l.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if type(self.pixmap_r) is QPixmap and self.active_frame is None:
            size = self.pixmap_r.size() * min(self.width() * (1 - self.separator_line) / self.pixmap_r.width(), self.height() / self.pixmap_r.height())
            # size *= self.img_zoom
            start_x = max(separator_x + 10, (self.width() - size.width()) // 2)
            start_point = QPoint(start_x, (self.height() - size.height()) // 2)

            draw_rect = QRect(start_point, size)
            self.painter.drawPixmap(draw_rect, self.pixmap_r.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def draw_dot(x, i, selected):
            color = [red_color, gray_color, green_color][self.suggest_marks[i] + 1]
            opacity = 80 if selected else 60
            self.painter.setBrush(QBrush(QColor(*color, opacity)))

            self.painter.setPen(black_solid_pen if selected else black_bold_pen)
            radius = 15 if selected else 25
            point_x = self.width() // 2 + x - radius
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
            if self.img_xy.y() > self.size().height() // 2:
                self.img_xy.setY(self.size().height() // 2)
            if self.img_xy.y() < -self.size().height() // 2:
                self.img_xy.setY(-self.size().height() // 2)
            # print(self.img_xy)
        elif self.adjustment_mode == 2:
            new_zoom = self.img_zoom * ((self.mouse_press_point.y() - event.y()) / 200 + 1)
            zoom_coef = new_zoom / self.img_zoom - 1
            xy_correction = QPoint(self.width() * (self.separator_line - .5) + 10, 10)
            self.img_xy += (self.img_xy + xy_correction / 2) * zoom_coef
            self.img_zoom = new_zoom
        elif self.adjustment_mode == 3:
            self.separator_line -= (self.mouse_press_point.x() - event.x()) / self.width()
            old_thumb_height = self.thumb_sheet.height()
            new_thumb_height = restrict(old_thumb_height + (self.mouse_press_point.y() - event.y()), 0, 1000) 
            self.thumb_sheet.setMaximumHeight(new_thumb_height)
            self.thumb_sheet.setMinimumHeight(new_thumb_height)
            self.thumb_sheet.setMaximumHeight(new_thumb_height)

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



class ThumbsView(QGraphicsView):
    view_left = 0
    last_mouse_x = 0
    last_mouse = QPoint()
    mouse_move_average = QPoint()
    mouse_direction_average = 1.
    autoscroll_speed = .0
    parking_stopper = .0
    this_tile_number = -1
    pressed_mouse = 0
    pressed_coord = QPoint()
    switch_category = 0
    mark = None
    show_pair = None
    scroll_stoper = 0
    center_tile = 0

    def __init__(self, scene: QGraphicsScene, parent: typing.Optional[QWidget]) -> None:
        QGraphicsView.__init__(self, scene, parent=parent)
        self.setMouseTracking(True)
        self.parent_sheet = parent
        self.last_mouse_x = self.size().width()

        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self.scroll_thumbs)
        self.scroll_timer.setInterval(30)
        self.scroll_timer.start()
        # self.scroll_timer

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self.parking_stopper = mix(self.parking_stopper, 1, .2)
        last_dxdy = event.pos() - self.last_mouse
        self.last_mouse = event.pos()
        last_direction = (restrict(abs(last_dxdy.x()) + .1, .1, 10) - restrict(abs(last_dxdy.y()) + .1, .1, 10)) / 8 + .5
        self.mouse_direction_average = mix(self.mouse_direction_average, last_direction, .1)

        # self.mouse_move_average = mix(self.mouse_move_average, event.pos() - self.last_mouse, .2)
        # scroll_down = restrict(abs(self.mouse_move_average.x()) + 1, 1, 10) / restrict(abs(self.mouse_move_average.y()) + 1, 1, 10)
        self.scroll_stoper = mix(0.5, self.scroll_stoper, smootherstep_ease(self.mouse_direction_average))
        # print(f"{self.mouse_direction_average:.3f}, {last_direction:.3f}, {self.scroll_stoper:.3f}")

        if self.pressed_mouse == 1:
            movement = event.pos() - self.pressed_coord
            movement_sum = abs(movement.x()) + abs(movement.y())
            if abs(movement.x()) < 30 and abs(movement.y()) < 30 and movement_sum < 45:
                pass
            elif abs(movement.x()) < abs(movement.y()):
                self.switch_category = 1 if movement.y() > 0 else 5                
            else: 
                self.switch_category = 4 if movement.x() > 0 else 2 
            # print(self.switch_category)
            return
        elif self.pressed_mouse != 0:
            return
        border = self.size().height() / 3
        closeness_to_border = smootherstep_ease(event.pos().x() / border)
        closeness_to_border *= smootherstep_ease(event.pos().y() / border)
        closeness_to_border *= smootherstep_ease((self.size().width() - event.pos().x()) / border)
        closeness_to_border *= smootherstep_ease((self.size().height() - event.pos().y()) / border)
        mouse_x_delta = event.pos().x() - self.last_mouse_x 
        self.last_mouse_x = event.pos().x()
        presision = smootherstep_ease(abs(self.autoscroll_speed) / 3 + .2) + .1
        self.autoscroll_speed = restrict(mix(self.autoscroll_speed, mouse_x_delta * closeness_to_border * presision, .1), -50, 50)

    def scroll_thumbs(self):
        self.parking_stopper = mix(self.parking_stopper, 0, .001)
        if self.scroll_stoper >= 0:
            self.scroll_stoper = mix(self.scroll_stoper, 1, .1)
        thumb_line_lenght = self.parent_sheet.pair_count * self.size().height()
        if thumb_line_lenght > self.size().width():
            desired_position = restrict(self.view_left, 0, thumb_line_lenght - self.size().width())
        else:
            desired_position = (thumb_line_lenght - self.size().width()) / 2
        if self.view_left != desired_position:
            parking_speed = smootherstep_ease(abs(self.view_left - desired_position) / self.size().width() * 3)
            self.view_left = mix(self.view_left, desired_position, parking_speed * (1 - self.parking_stopper))
            if self.autoscroll_speed * (self.view_left - desired_position) > 0:
                self.autoscroll_speed *= .1

        pair = self.parent_sheet.parent_object.current_pair
        thumb_size = self.parent_sheet.thumb_size
        distance_to_selected = pair * thumb_size - self.view_left - self.width() / 2
        if abs(distance_to_selected) < thumb_size:
            self.autoscroll_speed *= .9 * restrict(self.scroll_stoper, 0, 1)
        elif distance_to_selected * self.autoscroll_speed > 0:
            self.autoscroll_speed *= .99
            if abs(self.autoscroll_speed) > abs(distance_to_selected) / 3000:
                self.autoscroll_speed = mix(self.autoscroll_speed, distance_to_selected / 2000, .2)
        else:
            self.autoscroll_speed *= .98
        self.view_left += self.autoscroll_speed * 30
        self.center_tile = (self.view_left + self.size().width() / 2) / self.parent_sheet.thumb_size
        self.setSceneRect(self.view_left, 0, self.size().width(), self.size().height())
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.this_tile_number = int((event.pos().x() + self.view_left) // self.size().height())
        self.scroll_timer.singleShot(300, Qt.TimerType.PreciseTimer, self.clean_fast_click)
        self.pressed_mouse |= int(event.button())
        self.pressed_coord = event.pos()
        self.switch_category = 0
        return super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.pressed_mouse &= ~ event.button()
        self.last_mouse_x = event.pos().x()
        new_pair = abs(self.this_tile_number) - self.parent_sheet.parent_object.current_pair
        if self.switch_category:
            marks = {1: (-1, -1),
                     2: (1, -1),
                     3: (0, 0),
                     4: (-1, 1),
                     5: (1, 1) }
            self.mark(new_pair, *marks[self.switch_category])
        elif self.this_tile_number in range(0, self.parent_sheet.pair_count):            
            self.parent_sheet.update_selection(abs(self.this_tile_number))
            self.parent_sheet.update()
            self.show_pair(new_pair, with_move=False)
        self.switch_category = 0
        self.scroll_stoper = -1
        self.scroll_timer.singleShot(400, Qt.TimerType.PreciseTimer, self.unstop_scroller)
        return super().mouseReleaseEvent(event)
    
    def clean_fast_click(self):
        self.this_tile_number = -abs(self.this_tile_number)
        if self.switch_category == 0:
            self.switch_category = 3

    def unstop_scroller(self):
        self.scroll_stoper = 0


class ThumbSheet_Scene(QWidget):
    pair_count = 0
    items_to_update = set()
    requested_thumbs = set()
    thumbs_items = {}
    badges_version = 1

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.parent_object = parent
        self.setMouseTracking(True)
        self.thumb_size = restrict(self.size().height(), 30, 300)
        self.box_pen = QPen(Qt.GlobalColor.black, 3)
        self.load_pen = QPen(Qt.GlobalColor.yellow, 2)
        self.gray_pen = QPen(Qt.GlobalColor.gray, 1)

        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, self.size().width(), self.size().height())

        self.view = ThumbsView(self.scene, parent=self)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.vbox = QHBoxLayout(self)
        self.vbox.addWidget(self.view)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        
        self.preload_timer = QTimer()
        self.preload_timer.timeout.connect(self.thumb_requester)
        self.preload_timer.setInterval(50)

        self.clear_scene()

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.thumb_size = restrict(a0.size().height(), 30, 300)
        self.scene.setSceneRect(self.view.view_left, 0, a0.size().width(), a0.size().height())
        self.view.last_mouse_x = self.size().width() / 2
        
        self.update_all_thumbs()
        self.update_selection()
        return super().resizeEvent(a0)

    def update_all_thumbs(self):
        self.items_to_update = self.items_to_update.union(self.thumbs_items.keys())
        for thm in self.thumbs_items.values():
            thm.set_box(self.gray_pen)
        self.update()

    def update_selection(self, pair=None, with_move=False):
        if pair is None:
            pair = self.parent_object.current_pair
        self.rect.setRect(pair * self.thumb_size, 3, self.thumb_size - 6, self.thumb_size - 6)
        if with_move:
            self.view.autoscroll_speed = .00100 * (pair * self.thumb_size - self.view.view_left - self.view.width() / 2)

    def add_thumb_item(self, id, pixmap):
        new_pixmap_item = self.thumbs_items.get(id, ImageThumb(id, self.scene))
        if pixmap:
            new_pixmap_item.setPixmap(pixmap)
            new_pixmap_item.set_box(self.box_pen, self.thumb_size)
            self.requested_thumbs.discard(id)
            self.parent_object.update_single_thumb_badges(id)
        else:
            new_pixmap_item.set_box(self.load_pen, self.thumb_size)
        self.thumbs_items[id] = new_pixmap_item
        
    def thumb_requester(self):

        if thumb_queue:
            id, thumb_img = thumb_queue.popitem()
            pixmap = pixmap_from_image(thumb_img)
            self.add_thumb_item(id, pixmap)
        
        if len(self.requested_thumbs) > 10:
            return
        
        preload_count = 3 * self.size().width() // self.thumb_size
        for i in range(1, preload_count):
            thumb_id = int(self.view.center_tile) - ((-i, i)[i % 2] // 2)
            if thumb_id not in range(0, self.pair_count):
                continue
            
            if thumb_id not in self.items_to_update:
                if thumb_id in self.thumbs_items.keys():
                    if self.thumbs_items[thumb_id].badges_version != self.badges_version:
                        self.parent_object.update_single_thumb_badges(thumb_id)
                    continue
                if thumb_id in self.requested_thumbs:
                    continue

            self.items_to_update.discard(thumb_id)
            self.parent_object.request_thumb(thumb_id)
            self.requested_thumbs.add(thumb_id)
            self.add_thumb_item(thumb_id, None)
            # print(f"Pair {thumb_id} stage -")
            return
    
    def update_badges(self, thumb_id, mark_set):
        thumb_item = self.thumbs_items.get(thumb_id, None)
        if thumb_item:
            thumb_item.set_marks_ext(mark_set, self.badges_version, self.parent_object.current_pair)

    def clear_scene(self, new_pair_count=0):
        self.rect = QGraphicsRectItem(3, 3, self.thumb_size - 6, self.thumb_size - 6)
        self.rect.setPen(QPen(Qt.GlobalColor.red, 3))
        self.rect.setZValue(10)
        
        self.pair_count = new_pair_count
        self.items_to_update.clear()
        self.requested_thumbs.clear()
        self.thumbs_items.clear()
        self.scene.clear()
        self.scene.addItem(self.rect)
    
    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        self.preload_timer.start()
        return super().showEvent(a0)

    def hideEvent(self, a0: QtGui.QHideEvent) -> None:
        self.preload_timer.stop()
        return super().hideEvent(a0)


class ImageThumb(QtWidgets.QGraphicsPixmapItem):
    badges_version = 0

    def __init__(self, thumb_id, parent_scene: QGraphicsScene):
        QtWidgets.QGraphicsPixmapItem.__init__(self, parent=None)
        self.red_brush = QBrush(Qt.red)
        self.red_brush_semi = QBrush(QtGui.QColor(255, 0, 0, 50))
        self.green_brush = QBrush(Qt.green)
        self.green_brush_semi = QBrush(QtGui.QColor(0, 255, 0, 50))
        self.brush_list = [self.red_brush, self.red_brush_semi,
                           self.green_brush, self.green_brush_semi]
        self.black_pen = QPen(Qt.GlobalColor.black, 1)

        def init_element(go_green=False):
            new_item = QGraphicsEllipseItem(parent=self)
            new_item.setBrush(self.green_brush if go_green else self.red_brush)
            new_item.setPen(self.black_pen)
            new_item.hide()
            return new_item

        self.box = QGraphicsRectItem(parent=self)

        self.left_suggest = init_element()
        self.right_suggest = init_element()
        self.left_fatal = init_element()
        self.right_fatal = init_element()

        self.left_choose_plus = init_element(True)
        self.left_choose_minus = init_element()
        self.right_choose_plus = init_element(True)
        self.right_choose_minus = init_element()

        self.thumb_id = thumb_id
        parent_scene.addItem(self)
        self.setOffset(0, 3)
        self.moveBy(0, 3)

    def set_box(self, new_pen, thumb_size=None):
        self.box.setPen(new_pen)
        if thumb_size:
            self.box.setRect(0, 0, thumb_size - 6, thumb_size - 6)
            self.moveBy(thumb_size * self.thumb_id - self.x(), 0)

            self.adjust_rect(self.left_suggest, thumb_size, .2, .2, .1, .1)
            self.adjust_rect(self.right_suggest, thumb_size, .8, .2, .1, .1)
            self.adjust_rect(self.left_fatal, thumb_size, .2, .5, .2, .06)
            self.adjust_rect(self.right_fatal, thumb_size, .8, .5, .2, .06)

            self.adjust_rect(self.left_choose_plus, thumb_size, .2, .8, .06, .2)
            self.adjust_rect(self.left_choose_minus, thumb_size, .2, .8, .2, .06)
            self.adjust_rect(self.right_choose_plus, thumb_size, .8, .8, .06, .2)
            self.adjust_rect(self.right_choose_minus, thumb_size, .8, .8, .2, .06)

    def adjust_rect(self, obj: QGraphicsEllipseItem, thumb_size, cx: float, cy: float, w: float, h: float):
        left = (cx - w / 2) * thumb_size
        top = (cy - h / 2) * thumb_size
        obj.setRect(left, top, w * thumb_size, h * thumb_size)

    def set_marks_ext(self, marks, badges_version, current_pair):
        self.badges_version = badges_version
        dim_suggests = int(self.thumb_id > current_pair or any(marks[2:]))
        self.set_brush_pen(self.left_suggest, marks[0], dim_suggests)
        self.set_brush_pen(self.right_suggest, marks[1], dim_suggests)
        self.show_hide(self.left_fatal, marks[2])
        self.show_hide(self.right_fatal, marks[3])

        self.show_hide(self.left_choose_plus, marks[4] == 1)
        self.show_hide(self.left_choose_minus, marks[4] == -1)
        self.show_hide(self.right_choose_plus, marks[5] == 1)
        self.show_hide(self.right_choose_minus, marks[5] == -1)

    def show_hide(self, element, mark):
        element.show() if mark else element.hide()

    def set_brush_pen(self, item: QGraphicsEllipseItem, mark, translucent):
        if mark == 0:
            item.hide()
        else:
            item.show()
            brush_id = (mark > 0) * 2 + translucent
            item.setBrush(self.brush_list[brush_id])

