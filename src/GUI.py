# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import time
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QDialog
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QImage
import tkinter as tk
from tkinter import filedialog
# from src.LPLocator.LPLocator import LPLocator
import main

classes = {0: '0',
           1: '1',
           2: '2',
           3: '3',
           4: '4',
           5: '5',
           6: '6',
           7: '7',
           8: '8',
           9: '9',
           10: 'A',
           11: 'B',
           12: 'C',
           13: 'D',
           14: 'E',
           15: 'F',
           16: 'G',
           17: 'H',
           18: 'I',
           19: 'J',
           20: 'K',
           21: 'L',
           22: 'M',
           23: 'N',
           24: 'O',
           25: 'P',
           26: 'Q',
           27: 'R',
           28: 'S',
           29: 'T',
           30: 'U',
           31: 'V',
           32: 'W',
           33: 'X',
           34: 'Y',
           35: 'Z'}

class Ui_Form(object):

    # def setupUi(self, Form):
    #     Form.setObjectName("车牌号码识别")
    #     Form.resize(1050, 543)  # 793 543
    #     sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    #     sizePolicy.setHorizontalStretch(0)
    #     sizePolicy.setVerticalStretch(0)
    #     sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
    #     Form.setSizePolicy(sizePolicy)
    #     Form.setMinimumSize(QtCore.QSize(500, 500))
    #     self.gridLayout_2 = QtWidgets.QGridLayout(Form)
    #     self.gridLayout_2.setObjectName("gridLayout_2")
    #     self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
    #     self.horizontalLayout_2.setObjectName("horizontalLayout_2")
    #     self.gridLayout = QtWidgets.QGridLayout()
    #     self.gridLayout.setObjectName("gridLayout")
    #     spacerItem = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
    #     self.gridLayout.addItem(spacerItem, 2, 2, 1, 1)
    #     spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    #     self.gridLayout.addItem(spacerItem1, 2, 4, 1, 1)
    #     spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    #     self.gridLayout.addItem(spacerItem2, 2, 0, 1, 1)
    #     self.Button_select = QtWidgets.QPushButton(Form)
    #     self.Button_select.setEnabled(True)
    #     self.Button_select.setObjectName("Button_select")
    #     self.Button_select.setFixedSize(100, 30)
    #     self.gridLayout.addWidget(self.Button_select, 2, 1, 1, 1)
    #     self.label_phote = QtWidgets.QLabel(Form)
    #     self.label_phote.setEnabled(True)
    #     sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    #     sizePolicy.setHorizontalStretch(0)
    #     sizePolicy.setVerticalStretch(0)
    #     sizePolicy.setHeightForWidth(self.label_phote.sizePolicy().hasHeightForWidth())
    #     self.label_phote.setSizePolicy(sizePolicy)
    #     self.label_phote.setMinimumSize(QtCore.QSize(400, 400))
    #     self.label_phote.setObjectName("label_phote")
    #     self.gridLayout.addWidget(self.label_phote, 0, 0, 1, 5)
    #     self.Button_exit = QtWidgets.QPushButton(Form)
    #     self.Button_exit.setObjectName("Button_exit")
    #     self.gridLayout.addWidget(self.Button_exit, 2, 3, 1, 1)
    #     self.Button_exit.setFixedSize(100, 30)
    #     spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
    #     self.gridLayout.addItem(spacerItem3, 1, 2, 1, 1)
    #     self.horizontalLayout_2.addLayout(self.gridLayout)
    #     self.verticalLayout = QtWidgets.QVBoxLayout()
    #     self.verticalLayout.setObjectName("verticalLayout")
    #     self.label_located = QtWidgets.QLabel(Form)
    #     self.label_located.setObjectName("label_located")
    #     self.verticalLayout.addWidget(self.label_located)
    #     self.label_projection = QtWidgets.QLabel(Form)
    #     self.label_projection.setObjectName("label_projection")
    #     self.verticalLayout.addWidget(self.label_projection)
    #     self.label_division = QtWidgets.QLabel(Form)
    #     self.label_division.setObjectName("label_division")
    #     self.verticalLayout.addWidget(self.label_division)
    #     self.horizontalLayout = QtWidgets.QHBoxLayout()
    #     self.horizontalLayout.setObjectName("horizontalLayout")
    #     self.label_showresult = QtWidgets.QLabel(Form)
    #     font = QtGui.QFont()
    #     font.setFamily("Microsoft YaHei")
    #     font.setPointSize(14)
    #     self.label_showresult.setFont(font)
    #     self.label_showresult.setObjectName("label_showresult")
    #     self.horizontalLayout.addWidget(self.label_showresult)
    #     self.lineEdit_result = QtWidgets.QLineEdit(Form)
    #     font = QtGui.QFont()
    #     font.setFamily("Microsoft YaHei")
    #     font.setPointSize(18)
    #     font.setBold(False)
    #     font.setWeight(50)
    #     self.lineEdit_result.setFont(font)
    #     self.lineEdit_result.setObjectName("lineEdit_result")
    #     self.horizontalLayout.addWidget(self.lineEdit_result)
    #     spacerItem4 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
    #     self.horizontalLayout.addItem(spacerItem4)
    #     self.verticalLayout.addLayout(self.horizontalLayout)
    #     self.horizontalLayout_2.addLayout(self.verticalLayout)
    #     self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
    #
    #     self.retranslateUi(Form)
    #     QtCore.QMetaObject.connectSlotsByName(Form)
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1399, 840)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(1080, 654))
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem, 0, 1, 1, 2)
        self.tabWidget = QtWidgets.QTabWidget(Form)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setAutoFillBackground(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setAutoFillBackground(True)
        self.tab_1.setObjectName("tab_1")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_1)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.tab_1)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 1, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 4, 5, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 4, 1, 1, 1)
        self.Button_exit = QtWidgets.QPushButton(self.tab_1)
        self.Button_exit.setMinimumSize(QtCore.QSize(150, 40))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.Button_exit.setFont(font)
        self.Button_exit.setStyleSheet("")
        self.Button_exit.setObjectName("Button_exit")
        self.gridLayout.addWidget(self.Button_exit, 4, 4, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 4, 3, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem4, 0, 1, 1, 5)
        self.scrollArea = QtWidgets.QScrollArea(self.tab_1)
        self.scrollArea.setMinimumSize(QtCore.QSize(500, 530))
        self.scrollArea.setStyleSheet("border:2.9px solid rgb(205, 205, 205);\n"
                                      "border-radius:10px;")
        self.scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollArea.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scrollArea.setLineWidth(0)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 757, 616))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_phote = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_phote.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_phote.sizePolicy().hasHeightForWidth())
        self.label_phote.setSizePolicy(sizePolicy)
        self.label_phote.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_phote.setFont(font)
        self.label_phote.setStyleSheet("")
        self.label_phote.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_phote.setLineWidth(0)
        self.label_phote.setAlignment(QtCore.Qt.AlignCenter)
        self.label_phote.setObjectName("label_phote")
        self.gridLayout_4.addWidget(self.label_phote, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 2, 1, 1, 5)
        self.Button_select = QtWidgets.QPushButton(self.tab_1)
        self.Button_select.setEnabled(True)
        self.Button_select.setMinimumSize(QtCore.QSize(150, 40))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.Button_select.setFont(font)
        self.Button_select.setStyleSheet("")
        self.Button_select.setObjectName("Button_select")
        self.gridLayout.addWidget(self.Button_select, 4, 2, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem5, 3, 1, 1, 5)
        self.horizontalLayout_2.addLayout(self.gridLayout)
        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem6)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem7 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem7)
        self.label_2 = QtWidgets.QLabel(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.label_located = QtWidgets.QLabel(self.tab_1)
        self.label_located.setMinimumSize(QtCore.QSize(0, 150))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_located.setFont(font)
        self.label_located.setStyleSheet("border:5px solid rgb(205, 205, 205);\n"
                                         "border-radius:10px;")
        self.label_located.setAlignment(QtCore.Qt.AlignCenter)
        self.label_located.setObjectName("label_located")
        self.verticalLayout.addWidget(self.label_located)
        spacerItem8 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem8)
        self.label_3 = QtWidgets.QLabel(self.tab_1)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_projection = QtWidgets.QLabel(self.tab_1)
        self.label_projection.setMinimumSize(QtCore.QSize(0, 150))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_projection.setFont(font)
        self.label_projection.setStyleSheet("border:5px solid rgb(205, 205, 205);\n"
                                            "border-radius:10px;")
        self.label_projection.setAlignment(QtCore.Qt.AlignCenter)
        self.label_projection.setObjectName("label_projection")
        self.verticalLayout.addWidget(self.label_projection)
        spacerItem9 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem9)
        self.label_4 = QtWidgets.QLabel(self.tab_1)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.label_division = QtWidgets.QLabel(self.tab_1)
        self.label_division.setMinimumSize(QtCore.QSize(0, 150))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_division.setFont(font)
        self.label_division.setStyleSheet("border:5px solid rgb(205, 205, 205);\n"
                                          "border-radius:10px;")
        self.label_division.setAlignment(QtCore.Qt.AlignCenter)
        self.label_division.setObjectName("label_division")
        self.verticalLayout.addWidget(self.label_division)
        spacerItem10 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem10)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_showresult = QtWidgets.QLabel(self.tab_1)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label_showresult.setFont(font)
        self.label_showresult.setObjectName("label_showresult")
        self.horizontalLayout.addWidget(self.label_showresult)
        spacerItem11 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem11)
        self.lineEdit_result = QtWidgets.QLineEdit(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_result.sizePolicy().hasHeightForWidth())
        self.lineEdit_result.setSizePolicy(sizePolicy)
        self.lineEdit_result.setMinimumSize(QtCore.QSize(250, 50))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_result.setFont(font)
        self.lineEdit_result.setStyleSheet("border:5px solid rgb(205, 205, 205);\n"
                                           "border-radius:10px;")
        self.lineEdit_result.setReadOnly(True)
        self.lineEdit_result.setObjectName("lineEdit_result")
        self.horizontalLayout.addWidget(self.lineEdit_result)
        spacerItem12 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem12)
        self.label_time = QtWidgets.QLabel(self.tab_1)
        self.label_time.setMinimumSize(QtCore.QSize(60, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_time.setFont(font)
        self.label_time.setStyleSheet("border:5px solid rgb(205, 205, 205);\n"
                                      "border-radius:10px;")
        self.label_time.setObjectName("label_time")
        self.horizontalLayout.addWidget(self.label_time)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout_5.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setAutoFillBackground(True)
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.tableWidget = QtWidgets.QTableWidget(self.tab_2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(12)

        for i in range(12):
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setVerticalHeaderItem(i, item)
            for j in range(4):
                if j == 0:
                    item = QtWidgets.QTableWidgetItem()
                    self.tableWidget.setHorizontalHeaderItem(i, item)
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, j, item)

        self.tableWidget.horizontalHeader().setDefaultSectionSize(150)

        self.gridLayout_3.addWidget(self.tableWidget, 5, 2, 1, 1)
        self.table_matrix = QtWidgets.QTableWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.table_matrix.sizePolicy().hasHeightForWidth())
        self.table_matrix.setSizePolicy(sizePolicy)
        self.table_matrix.setMinimumSize(QtCore.QSize(700, 450))
        self.table_matrix.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_matrix.setColumnCount(36)
        self.table_matrix.setObjectName("table_matrix")
        self.table_matrix.setRowCount(36)

        for i in range(36):
            item = QtWidgets.QTableWidgetItem()
            self.table_matrix.setVerticalHeaderItem(i, item)

            item = QtWidgets.QTableWidgetItem()
            self.table_matrix.setHorizontalHeaderItem(i, item)

            for j in range(36):
                item = QtWidgets.QTableWidgetItem()
                self.table_matrix.setItem(i, j, item)

            brush = QtGui.QBrush(QtGui.QColor(50, 255, 80))
            brush.setStyle(QtCore.Qt.SolidPattern)
            self.table_matrix.item(i, i).setBackground(brush)


        self.table_matrix.horizontalHeader().setDefaultSectionSize(30)
        self.table_matrix.verticalHeader().setDefaultSectionSize(30)
        self.gridLayout_3.addWidget(self.table_matrix, 2, 0, 4, 1)
        self.table_indicator = QtWidgets.QTableWidget(self.tab_2)
        self.table_indicator.setObjectName("table_indicator")
        self.table_indicator.setColumnCount(5)
        self.table_indicator.setRowCount(36)
        for i in range(36):
            item = QtWidgets.QTableWidgetItem()
            self.table_indicator.setVerticalHeaderItem(i, item)
            for j in range(5):
                item = QtWidgets.QTableWidgetItem()
                self.table_indicator.setItem(i, j, item)
        for i in range(5):
            item = QtWidgets.QTableWidgetItem()
            self.table_indicator.setHorizontalHeaderItem(i, item)

        self.table_indicator.horizontalHeader().setDefaultSectionSize(105)
        self.gridLayout_3.addWidget(self.table_indicator, 2, 2, 1, 1)
        spacerItem12 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem12, 3, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 0, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 4, 2, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem13, 2, 1, 4, 1)
        self.gridLayout_6.addLayout(self.gridLayout_3, 0, 0, 1, 2)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_2.addWidget(self.tabWidget, 1, 1, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem14, 1, 0, 4, 1)
        spacerItem15 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem15, 1, 3, 4, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        self.Button_select.clicked['bool'].connect(self.label_projection.setDisabled)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Form)
    # def retranslateUi(self, Form):
    #     _translate = QtCore.QCoreApplication.translate
    #     Form.setWindowTitle(_translate("Form", "车牌号码识别"))
    #     self.Button_select.setText(_translate("Form", "选择识别图片"))
    #     self.label_phote.setText(_translate("Form", "选择的图片"))
    #     self.Button_exit.setText(_translate("Form", "退出"))
    #     self.label_located.setText(_translate("Form", "车牌定位图"))
    #     self.label_projection.setText(_translate("Form", "垂直投影图"))
    #     self.label_division.setText(_translate("Form", "车牌分割图"))
    #     self.label_showresult.setText(_translate("Form", "识别结果："))

        # self.retranslateUi(Form)
        # # self.Button_select.clicked['bool'].connect(self.label_projection.setDisabled)
        # QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "原始图片"))
        self.Button_exit.setText(_translate("Form", "退出应用"))
        self.label_phote.setText(_translate("Form", "(当前未选择图片)"))
        self.Button_select.setText(_translate("Form", "选择识别图片"))
        self.label_2.setText(_translate("Form", "车牌定位"))
        self.label_located.setText(_translate("Form", "(等待图片导入)"))
        self.label_3.setText(_translate("Form", "垂直投影处理"))
        self.label_projection.setText(_translate("Form", "(等待图片导入)"))
        self.label_4.setText(_translate("Form", "字符分割示意图"))
        self.label_division.setText(_translate("Form", "(等待图片导入)"))
        self.label_showresult.setText(_translate("Form", "识别结果："))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("Form", "识别演示"))
        for i in range(9, 12):
            item = self.tableWidget.verticalHeaderItem(i)
            item.setText(_translate("Form", "*"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "Layer"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Form", "Layer Type"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Form", "Output Shape"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Form", "Param"))

        content = [['conv2d_1', 'Conv2D', '(138, 43, 128)', '1280'],
                   ['max_pooling2d_1', 'MaxPooling2D', '(34, 10, 128)', '0'],
                   ['conv2d_2', 'Conv2D', '(32, 8, 68)', '78404'],
                   ['max_pooling2d_2', 'MaxPooling2D', '(16, 4, 68)', '0'],
                   ['flatten_1', 'Flatten', '(4352)', '0'],
                   ['dense_1', 'Dense', '(128)', '557184'],
                   ['dropout_1', 'Dropout', '(128)', '0'],
                   ['dense_2', 'Dense', '(68)', '8872'],
                   ['', '', '', ''],
                   ['Total params', '645640', '', ''],
                   ['Trainable params', '645640', '', ''],
                   ['None-trainable params', '0', '', '']]
        for i in range(12):
            for j in range(4):
                item = self.tableWidget.item(i, j)
                item.setText(_translate("Form", content[i][j]))

        head = ['0','1','2','3','4','5','6','7','8','9',
                'A','B','C','D','E','F','G','H','I','J',
                'K','L','M','N','O','P','Q','R','S','T',
                'U','V','W','X','Y','Z']
        for i in range(36):
            item = self.table_matrix.verticalHeaderItem(i)
            item.setText(_translate("Form", head[i]))

            item = self.table_matrix.horizontalHeaderItem(i)
            item.setText(_translate("Form", head[i]))

        __sortingEnabled = self.table_matrix.isSortingEnabled()
        self.table_matrix.setSortingEnabled(False)
        self.table_matrix.setSortingEnabled(__sortingEnabled)
        for i in range(36):
            item = self.table_indicator.verticalHeaderItem(i)
            item.setText(_translate("Form", ""))

        item = self.table_indicator.horizontalHeaderItem(0)
        item.setText(_translate("Form", "类"))
        item = self.table_indicator.horizontalHeaderItem(1)
        item.setText(_translate("Form", "Accuracy"))
        item = self.table_indicator.horizontalHeaderItem(2)
        item.setText(_translate("Form", "Precision"))
        item = self.table_indicator.horizontalHeaderItem(3)
        item.setText(_translate("Form", "Recall"))
        item = self.table_indicator.horizontalHeaderItem(4)
        item.setText(_translate("Form", "F1_score"))
        self.label_6.setText(_translate("Form", "分类指标"))
        self.label_5.setText(_translate("Form", "混淆矩阵"))
        self.label_7.setText(_translate("Form", "模型信息"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "模型概况"))

        self.Button_exit.clicked['bool'].connect(self.quit)
        self.Button_select.clicked['bool'].connect(self.select_picture)

    def quit(self):
        print("1")
        sys.exit()

    def select_picture(self):
        root = tk.Tk()
        root.withdraw()
        # Folderpath = filedialog.askdirectory()  # 获得选择好的文件夹
        Filepath = filedialog.askopenfilename(
            title='请选择要用于车牌识别的图像',
            filetypes=[('JPG', '*.jpg *.jpeg'), ('PNG', '*.png')])  # 获得选择好的文件

        print('Filepath:', Filepath)
        img = QImage(Filepath)  # 创建图片实例

        if img is None or img.width() * img.height() <= 0:
            self.lineEdit_result.setText("请选择正确的图片路径")
            return
        # 图片按窗口大小比例缩放
        ratio = (self.scrollArea.width() - 70) / img.width()
        newWidth = int(ratio * img.width())
        newHeight = int(ratio * img.height())
        size = QSize(newWidth, newHeight)
        pixImg = QPixmap.fromImage(img.scaled(size, Qt.IgnoreAspectRatio))
        # 修改图片实例大小并从QImage实例中生成QPixmap实例以备放入QLabel控件中
        self.label_phote.resize(newWidth, newHeight)
        self.label_phote.setPixmap(pixImg)
        # print('Folderpath:', Folderpath)

        time_start_1 = time.time()
        img_lp_highlighted, shadow_image, sliced_photos, result = main.Main(Filepath)
        time_end_1 = time.time()
        # >>>>>>>>>>> 2022-11-14 运行时间
        print("运行时间：" + str(time_end_1 - time_start_1) + "秒")
        img_located = img_lp_highlighted
        shrink = cv2.cvtColor(img_located, cv2.COLOR_BGR2RGB)
        img_located = QtGui.QImage(shrink.data,
                                   shrink.shape[1],
                                   shrink.shape[0],
                                   shrink.shape[1] * 3,
                                   QtGui.QImage.Format_BGR888)

        newWidth_located = int(430)
        newHeight_located = int(300)
        size_located = QSize(newWidth_located, newHeight_located)
        self.label_phote.resize(newWidth, newHeight)
        self.label_located.setPixmap(QtGui.QPixmap.fromImage(img_located.scaled(size_located, Qt.IgnoreAspectRatio)))

        shrink = cv2.cvtColor(sliced_photos, cv2.COLOR_BGR2RGB)
        sliced_photos = QtGui.QImage(shrink.data,
                                     shrink.shape[1],
                                     shrink.shape[0],
                                     shrink.shape[1] * 3,
                                     QtGui.QImage.Format_BGR888)
        self.label_division.setPixmap(QtGui.QPixmap.fromImage(sliced_photos))

        shrink = cv2.cvtColor(shadow_image, cv2.COLOR_BGR2RGB)
        shadow_image = QtGui.QImage(shrink.data,
                                    shrink.shape[1],
                                    shrink.shape[0],
                                    shrink.shape[1] * 3,
                                    QtGui.QImage.Format_BGR888)
        self.label_projection.setPixmap(QtGui.QPixmap.fromImage(shadow_image))

        self.lineEdit_result.setText(result)
        time_end_1 = time.time()
        time_str = str(time_end_1 - time_start_1)[:4]
        self.label_time.setText(time_str)
        print('Sliced photos:', sliced_photos)

    def fill_tables(self):  # TODO UNTEST
        confusion = main.en_identifier.confusion
        evaluation = main.en_identifier.evaluation
        if len(confusion) == 0 or len(evaluation) == 0: return
        for i in range(36):
            for j in range(36):
                content = str(int(confusion[i][j]))
                if confusion[i][j] == 0:    content = ''
                elif i != j:
                    brush = QtGui.QBrush(QtGui.QColor(200, 255, 210))
                    brush.setStyle(QtCore.Qt.SolidPattern)
                    self.table_matrix.item(i, j).setBackground(brush)

                self.table_matrix.item(i, j).setText(content)
                self.table_matrix.item(i, j).setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            for j in range(5):
                if j == 0:
                    self.table_indicator.item(i, j).setText(classes[i])
                else:
                    self.table_indicator.item(i, j).setText(str('%.3f' % evaluation[i][j-1]))
                self.table_indicator.item(i, j).setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            if evaluation[i][1] == 1:
                brush = QtGui.QBrush(QtGui.QColor(50, 255, 80))
                brush.setStyle(QtCore.Qt.SolidPattern)
                self.table_indicator.item(i, 0).setBackground(brush)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = QWidget()
    ui = Ui_Form()
    ui.setupUi(mainWindow)
    ui.fill_tables()
    mainWindow.show()
    sys.exit(app.exec_())
