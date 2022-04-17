# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CompareDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1261, 890)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(95, 95, 95))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        MainWindow.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color:rgb(95, 95, 95);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(600, 400))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout.setSpacing(4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_img = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_img.sizePolicy().hasHeightForWidth())
        self.frame_img.setSizePolicy(sizePolicy)
        self.frame_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_img.setMidLineWidth(-1)
        self.frame_img.setObjectName("frame_img")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_img)
        self.verticalLayout_2.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout.addWidget(self.frame_img)
        self.frame_down = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_down.sizePolicy().hasHeightForWidth())
        self.frame_down.setSizePolicy(sizePolicy)
        self.frame_down.setMinimumSize(QtCore.QSize(0, 120))
        self.frame_down.setStyleSheet("background-color:rgb(195, 195, 195);")
        self.frame_down.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_down.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_down.setObjectName("frame_down")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_down)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_percent = QtWidgets.QLabel(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_percent.sizePolicy().hasHeightForWidth())
        self.label_percent.setSizePolicy(sizePolicy)
        self.label_percent.setMinimumSize(QtCore.QSize(160, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_percent.setFont(font)
        self.label_percent.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.label_percent.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_percent.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_percent.setLineWidth(3)
        self.label_percent.setAlignment(QtCore.Qt.AlignCenter)
        self.label_percent.setIndent(-13)
        self.label_percent.setObjectName("label_percent")
        self.horizontalLayout.addWidget(self.label_percent)
        self.frame_1 = QtWidgets.QFrame(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_1.sizePolicy().hasHeightForWidth())
        self.frame_1.setSizePolicy(sizePolicy)
        self.frame_1.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_1.setLineWidth(2)
        self.frame_1.setMidLineWidth(0)
        self.frame_1.setObjectName("frame_1")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_1)
        self.verticalLayout_4.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout_4.setSpacing(2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_6 = QtWidgets.QLabel(self.frame_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_4.addWidget(self.label_6)
        self.line = QtWidgets.QFrame(self.frame_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy)
        self.line.setMinimumSize(QtCore.QSize(10, 0))
        self.line.setMidLineWidth(0)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_4.addWidget(self.line)
        self.push_bg_black = QtWidgets.QPushButton(self.frame_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_bg_black.sizePolicy().hasHeightForWidth())
        self.push_bg_black.setSizePolicy(sizePolicy)
        self.push_bg_black.setMinimumSize(QtCore.QSize(30, 0))
        self.push_bg_black.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_bg_black.setStyleSheet("background-color: rgb(40,40,40);")
        self.push_bg_black.setText("")
        self.push_bg_black.setCheckable(True)
        self.push_bg_black.setChecked(True)
        self.push_bg_black.setObjectName("push_bg_black")
        self.mode_buttons = QtWidgets.QButtonGroup(MainWindow)
        self.mode_buttons.setObjectName("mode_buttons")
        self.mode_buttons.addButton(self.push_bg_black)
        self.verticalLayout_4.addWidget(self.push_bg_black)
        self.push_bg_gray = QtWidgets.QPushButton(self.frame_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_bg_gray.sizePolicy().hasHeightForWidth())
        self.push_bg_gray.setSizePolicy(sizePolicy)
        self.push_bg_gray.setMinimumSize(QtCore.QSize(30, 0))
        self.push_bg_gray.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_bg_gray.setStyleSheet("background-color: rgb(128,128,128);")
        self.push_bg_gray.setText("")
        self.push_bg_gray.setCheckable(True)
        self.push_bg_gray.setChecked(False)
        self.push_bg_gray.setObjectName("push_bg_gray")
        self.mode_buttons.addButton(self.push_bg_gray)
        self.verticalLayout_4.addWidget(self.push_bg_gray)
        self.push_bg_white = QtWidgets.QPushButton(self.frame_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_bg_white.sizePolicy().hasHeightForWidth())
        self.push_bg_white.setSizePolicy(sizePolicy)
        self.push_bg_white.setMinimumSize(QtCore.QSize(30, 0))
        self.push_bg_white.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_bg_white.setStyleSheet("background-color: rgb(255,255,255);\n"
"")
        self.push_bg_white.setText("")
        self.push_bg_white.setIconSize(QtCore.QSize(12, 12))
        self.push_bg_white.setCheckable(True)
        self.push_bg_white.setObjectName("push_bg_white")
        self.mode_buttons.addButton(self.push_bg_white)
        self.verticalLayout_4.addWidget(self.push_bg_white)
        self.line_3 = QtWidgets.QFrame(self.frame_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_3.sizePolicy().hasHeightForWidth())
        self.line_3.setSizePolicy(sizePolicy)
        self.line_3.setMinimumSize(QtCore.QSize(10, 0))
        self.line_3.setMidLineWidth(0)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_4.addWidget(self.line_3)
        self.push_colored = QtWidgets.QPushButton(self.frame_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_colored.sizePolicy().hasHeightForWidth())
        self.push_colored.setSizePolicy(sizePolicy)
        self.push_colored.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_colored.setCheckable(True)
        self.push_colored.setChecked(True)
        self.push_colored.setObjectName("push_colored")
        self.verticalLayout_4.addWidget(self.push_colored)
        self.horizontalLayout.addWidget(self.frame_1)
        self.frame_2 = QtWidgets.QFrame(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setLineWidth(2)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_5.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout_5.setSpacing(2)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label = QtWidgets.QLabel(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_5.addWidget(self.label)
        self.line_5 = QtWidgets.QFrame(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_5.sizePolicy().hasHeightForWidth())
        self.line_5.setSizePolicy(sizePolicy)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout_5.addWidget(self.line_5)
        self.push_show_all = QtWidgets.QPushButton(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_show_all.sizePolicy().hasHeightForWidth())
        self.push_show_all.setSizePolicy(sizePolicy)
        self.push_show_all.setMinimumSize(QtCore.QSize(30, 0))
        self.push_show_all.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_show_all.setCheckable(True)
        self.push_show_all.setChecked(True)
        self.push_show_all.setObjectName("push_show_all")
        self.filter_buttons = QtWidgets.QButtonGroup(MainWindow)
        self.filter_buttons.setObjectName("filter_buttons")
        self.filter_buttons.addButton(self.push_show_all)
        self.verticalLayout_5.addWidget(self.push_show_all)
        self.push_show_identical = QtWidgets.QPushButton(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_show_identical.sizePolicy().hasHeightForWidth())
        self.push_show_identical.setSizePolicy(sizePolicy)
        self.push_show_identical.setMinimumSize(QtCore.QSize(30, 0))
        self.push_show_identical.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_show_identical.setCheckable(True)
        self.push_show_identical.setChecked(False)
        self.push_show_identical.setObjectName("push_show_identical")
        self.filter_buttons.addButton(self.push_show_identical)
        self.verticalLayout_5.addWidget(self.push_show_identical)
        self.push_show_size_and_crop = QtWidgets.QPushButton(self.frame_2)
        self.push_show_size_and_crop.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_show_size_and_crop.sizePolicy().hasHeightForWidth())
        self.push_show_size_and_crop.setSizePolicy(sizePolicy)
        self.push_show_size_and_crop.setMinimumSize(QtCore.QSize(30, 0))
        self.push_show_size_and_crop.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_show_size_and_crop.setCheckable(True)
        self.push_show_size_and_crop.setChecked(False)
        self.push_show_size_and_crop.setObjectName("push_show_size_and_crop")
        self.filter_buttons.addButton(self.push_show_size_and_crop)
        self.verticalLayout_5.addWidget(self.push_show_size_and_crop)
        self.push_show_crop_only = QtWidgets.QPushButton(self.frame_2)
        self.push_show_crop_only.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_show_crop_only.sizePolicy().hasHeightForWidth())
        self.push_show_crop_only.setSizePolicy(sizePolicy)
        self.push_show_crop_only.setMinimumSize(QtCore.QSize(30, 0))
        self.push_show_crop_only.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_show_crop_only.setCheckable(True)
        self.push_show_crop_only.setChecked(False)
        self.push_show_crop_only.setObjectName("push_show_crop_only")
        self.filter_buttons.addButton(self.push_show_crop_only)
        self.verticalLayout_5.addWidget(self.push_show_crop_only)
        self.push_show_size_only = QtWidgets.QPushButton(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_show_size_only.sizePolicy().hasHeightForWidth())
        self.push_show_size_only.setSizePolicy(sizePolicy)
        self.push_show_size_only.setMinimumSize(QtCore.QSize(30, 0))
        self.push_show_size_only.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_show_size_only.setIconSize(QtCore.QSize(12, 12))
        self.push_show_size_only.setCheckable(True)
        self.push_show_size_only.setObjectName("push_show_size_only")
        self.filter_buttons.addButton(self.push_show_size_only)
        self.verticalLayout_5.addWidget(self.push_show_size_only)
        self.push_show_mixed_only = QtWidgets.QPushButton(self.frame_2)
        self.push_show_mixed_only.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_show_mixed_only.sizePolicy().hasHeightForWidth())
        self.push_show_mixed_only.setSizePolicy(sizePolicy)
        self.push_show_mixed_only.setMinimumSize(QtCore.QSize(30, 0))
        self.push_show_mixed_only.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_show_mixed_only.setIconSize(QtCore.QSize(12, 12))
        self.push_show_mixed_only.setCheckable(True)
        self.push_show_mixed_only.setObjectName("push_show_mixed_only")
        self.filter_buttons.addButton(self.push_show_mixed_only)
        self.verticalLayout_5.addWidget(self.push_show_mixed_only)
        self.horizontalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setLineWidth(2)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_6.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout_6.setSpacing(2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_2 = QtWidgets.QLabel(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        self.line_6 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_6.sizePolicy().hasHeightForWidth())
        self.line_6.setSizePolicy(sizePolicy)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.verticalLayout_6.addWidget(self.line_6)
        self.push_suggest_size_and_crop = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_suggest_size_and_crop.sizePolicy().hasHeightForWidth())
        self.push_suggest_size_and_crop.setSizePolicy(sizePolicy)
        self.push_suggest_size_and_crop.setMinimumSize(QtCore.QSize(30, 0))
        self.push_suggest_size_and_crop.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_suggest_size_and_crop.setCheckable(True)
        self.push_suggest_size_and_crop.setChecked(True)
        self.push_suggest_size_and_crop.setObjectName("push_suggest_size_and_crop")
        self.suggest_buttons = QtWidgets.QButtonGroup(MainWindow)
        self.suggest_buttons.setObjectName("suggest_buttons")
        self.suggest_buttons.addButton(self.push_suggest_size_and_crop)
        self.verticalLayout_6.addWidget(self.push_suggest_size_and_crop)
        self.push_suggest_crop_only = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_suggest_crop_only.sizePolicy().hasHeightForWidth())
        self.push_suggest_crop_only.setSizePolicy(sizePolicy)
        self.push_suggest_crop_only.setMinimumSize(QtCore.QSize(30, 0))
        self.push_suggest_crop_only.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_suggest_crop_only.setCheckable(True)
        self.push_suggest_crop_only.setChecked(False)
        self.push_suggest_crop_only.setObjectName("push_suggest_crop_only")
        self.suggest_buttons.addButton(self.push_suggest_crop_only)
        self.verticalLayout_6.addWidget(self.push_suggest_crop_only)
        self.push_suggest_size_only = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_suggest_size_only.sizePolicy().hasHeightForWidth())
        self.push_suggest_size_only.setSizePolicy(sizePolicy)
        self.push_suggest_size_only.setMinimumSize(QtCore.QSize(30, 0))
        self.push_suggest_size_only.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_suggest_size_only.setCheckable(True)
        self.push_suggest_size_only.setChecked(False)
        self.push_suggest_size_only.setObjectName("push_suggest_size_only")
        self.suggest_buttons.addButton(self.push_suggest_size_only)
        self.verticalLayout_6.addWidget(self.push_suggest_size_only)
        self.push_suggest_any = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_suggest_any.sizePolicy().hasHeightForWidth())
        self.push_suggest_any.setSizePolicy(sizePolicy)
        self.push_suggest_any.setMinimumSize(QtCore.QSize(30, 0))
        self.push_suggest_any.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_suggest_any.setIconSize(QtCore.QSize(12, 12))
        self.push_suggest_any.setCheckable(True)
        self.push_suggest_any.setObjectName("push_suggest_any")
        self.suggest_buttons.addButton(self.push_suggest_any)
        self.verticalLayout_6.addWidget(self.push_suggest_any)
        self.push_suggest_filename = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_suggest_filename.sizePolicy().hasHeightForWidth())
        self.push_suggest_filename.setSizePolicy(sizePolicy)
        self.push_suggest_filename.setMinimumSize(QtCore.QSize(30, 0))
        self.push_suggest_filename.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_suggest_filename.setIconSize(QtCore.QSize(12, 12))
        self.push_suggest_filename.setCheckable(True)
        self.push_suggest_filename.setObjectName("push_suggest_filename")
        self.suggest_buttons.addButton(self.push_suggest_filename)
        self.verticalLayout_6.addWidget(self.push_suggest_filename)
        self.horizontalLayout.addWidget(self.frame_3)
        self.frame_4 = QtWidgets.QFrame(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setLineWidth(2)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_3.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout_3.setSpacing(2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_5 = QtWidgets.QLabel(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.line_9 = QtWidgets.QFrame(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_9.sizePolicy().hasHeightForWidth())
        self.line_9.setSizePolicy(sizePolicy)
        self.line_9.setMinimumSize(QtCore.QSize(10, 0))
        self.line_9.setMidLineWidth(0)
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.verticalLayout_3.addWidget(self.line_9)
        self.push_mark_suggested = QtWidgets.QPushButton(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_mark_suggested.sizePolicy().hasHeightForWidth())
        self.push_mark_suggested.setSizePolicy(sizePolicy)
        self.push_mark_suggested.setMinimumSize(QtCore.QSize(30, 0))
        self.push_mark_suggested.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_mark_suggested.setIconSize(QtCore.QSize(12, 12))
        self.push_mark_suggested.setCheckable(False)
        self.push_mark_suggested.setObjectName("push_mark_suggested")
        self.verticalLayout_3.addWidget(self.push_mark_suggested)
        self.push_apply_marked = QtWidgets.QPushButton(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_apply_marked.sizePolicy().hasHeightForWidth())
        self.push_apply_marked.setSizePolicy(sizePolicy)
        self.push_apply_marked.setMinimumSize(QtCore.QSize(30, 0))
        self.push_apply_marked.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_apply_marked.setIconSize(QtCore.QSize(12, 12))
        self.push_apply_marked.setCheckable(False)
        self.push_apply_marked.setObjectName("push_apply_marked")
        self.verticalLayout_3.addWidget(self.push_apply_marked)
        self.push_move_applied = QtWidgets.QPushButton(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_move_applied.sizePolicy().hasHeightForWidth())
        self.push_move_applied.setSizePolicy(sizePolicy)
        self.push_move_applied.setMinimumSize(QtCore.QSize(30, 0))
        self.push_move_applied.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_move_applied.setIconSize(QtCore.QSize(12, 12))
        self.push_move_applied.setCheckable(False)
        self.push_move_applied.setObjectName("push_move_applied")
        self.verticalLayout_3.addWidget(self.push_move_applied)
        self.line_4 = QtWidgets.QFrame(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_4.sizePolicy().hasHeightForWidth())
        self.line_4.setSizePolicy(sizePolicy)
        self.line_4.setMinimumSize(QtCore.QSize(10, 0))
        self.line_4.setMidLineWidth(0)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout_3.addWidget(self.line_4)
        self.push_mark_clear = QtWidgets.QPushButton(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_mark_clear.sizePolicy().hasHeightForWidth())
        self.push_mark_clear.setSizePolicy(sizePolicy)
        self.push_mark_clear.setMinimumSize(QtCore.QSize(30, 0))
        self.push_mark_clear.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_mark_clear.setIconSize(QtCore.QSize(12, 12))
        self.push_mark_clear.setCheckable(False)
        self.push_mark_clear.setObjectName("push_mark_clear")
        self.verticalLayout_3.addWidget(self.push_mark_clear)
        self.push_goto_first = QtWidgets.QPushButton(self.frame_4)
        self.push_goto_first.setObjectName("push_goto_first")
        self.verticalLayout_3.addWidget(self.push_goto_first)
        self.push_resort_pairs = QtWidgets.QPushButton(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.push_resort_pairs.sizePolicy().hasHeightForWidth())
        self.push_resort_pairs.setSizePolicy(sizePolicy)
        self.push_resort_pairs.setMinimumSize(QtCore.QSize(30, 0))
        self.push_resort_pairs.setFocusPolicy(QtCore.Qt.NoFocus)
        self.push_resort_pairs.setIconSize(QtCore.QSize(12, 12))
        self.push_resort_pairs.setCheckable(False)
        self.push_resort_pairs.setObjectName("push_resort_pairs")
        self.verticalLayout_3.addWidget(self.push_resort_pairs)
        self.horizontalLayout.addWidget(self.frame_4)
        self.label_name_l = QtWidgets.QLabel(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_name_l.sizePolicy().hasHeightForWidth())
        self.label_name_l.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_name_l.setFont(font)
        self.label_name_l.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_name_l.setText("")
        self.label_name_l.setAlignment(QtCore.Qt.AlignCenter)
        self.label_name_l.setWordWrap(True)
        self.label_name_l.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label_name_l.setObjectName("label_name_l")
        self.horizontalLayout.addWidget(self.label_name_l)
        self.label_size_l = QtWidgets.QLabel(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_size_l.sizePolicy().hasHeightForWidth())
        self.label_size_l.setSizePolicy(sizePolicy)
        self.label_size_l.setMinimumSize(QtCore.QSize(150, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_size_l.setFont(font)
        self.label_size_l.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_size_l.setText("")
        self.label_size_l.setAlignment(QtCore.Qt.AlignCenter)
        self.label_size_l.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label_size_l.setObjectName("label_size_l")
        self.horizontalLayout.addWidget(self.label_size_l)
        self.label_size_r = QtWidgets.QLabel(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_size_r.sizePolicy().hasHeightForWidth())
        self.label_size_r.setSizePolicy(sizePolicy)
        self.label_size_r.setMinimumSize(QtCore.QSize(150, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_size_r.setFont(font)
        self.label_size_r.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_size_r.setText("")
        self.label_size_r.setAlignment(QtCore.Qt.AlignCenter)
        self.label_size_r.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label_size_r.setObjectName("label_size_r")
        self.horizontalLayout.addWidget(self.label_size_r)
        self.label_name_r = QtWidgets.QLabel(self.frame_down)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_name_r.sizePolicy().hasHeightForWidth())
        self.label_name_r.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_name_r.setFont(font)
        self.label_name_r.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_name_r.setText("")
        self.label_name_r.setAlignment(QtCore.Qt.AlignCenter)
        self.label_name_r.setWordWrap(True)
        self.label_name_r.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label_name_r.setObjectName("label_name_r")
        self.horizontalLayout.addWidget(self.label_name_r)
        self.verticalLayout.addWidget(self.frame_down)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Compare images"))
        self.label_percent.setText(_translate("MainWindow", "Pair info"))
        self.label_6.setText(_translate("MainWindow", "Previews"))
        self.push_colored.setText(_translate("MainWindow", "Colored"))
        self.label.setText(_translate("MainWindow", "Filter pairs "))
        self.push_show_all.setText(_translate("MainWindow", "All"))
        self.push_show_identical.setText(_translate("MainWindow", "Identical"))
        self.push_show_size_and_crop.setText(_translate("MainWindow", "Size && crop"))
        self.push_show_crop_only.setText(_translate("MainWindow", "Crop only"))
        self.push_show_size_only.setText(_translate("MainWindow", "Size only"))
        self.push_show_mixed_only.setText(_translate("MainWindow", "Mixed only"))
        self.label_2.setText(_translate("MainWindow", " Suggest "))
        self.push_suggest_size_and_crop.setText(_translate("MainWindow", "Size && crop"))
        self.push_suggest_crop_only.setText(_translate("MainWindow", "Crop only"))
        self.push_suggest_size_only.setText(_translate("MainWindow", "Size only"))
        self.push_suggest_any.setText(_translate("MainWindow", "Any image"))
        self.push_suggest_filename.setText(_translate("MainWindow", "Filename"))
        self.label_5.setText(_translate("MainWindow", "Actions"))
        self.push_mark_suggested.setText(_translate("MainWindow", " Mark suggested "))
        self.push_apply_marked.setText(_translate("MainWindow", "Apply marked"))
        self.push_move_applied.setText(_translate("MainWindow", "Move applied"))
        self.push_mark_clear.setText(_translate("MainWindow", "Clear marks"))
        self.push_goto_first.setText(_translate("MainWindow", "Go to first pair"))
        self.push_resort_pairs.setText(_translate("MainWindow", "Resort pairs"))
