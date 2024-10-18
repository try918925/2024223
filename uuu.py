# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'new.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QTextEdit, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(1278, 344))
        self.label.setStyleSheet(u"border: 2px solid red;")

        self.verticalLayout.addWidget(self.label)

        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMaximumSize(QSize(1278, 344))
        self.label_2.setStyleSheet(u"border: 2px solid red;")

        self.verticalLayout.addWidget(self.label_2)

        self.label_3 = QLabel(self.widget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMaximumSize(QSize(1278, 16777215))
        self.label_3.setStyleSheet(u"border: 2px solid red;")

        self.verticalLayout.addWidget(self.label_3)


        self.gridLayout.addWidget(self.widget, 0, 0, 1, 1)

        self.widget_2 = QWidget(self.centralwidget)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setMaximumSize(QSize(600, 16777215))
        self.verticalLayout_2 = QVBoxLayout(self.widget_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_4 = QLabel(self.widget_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(582, 344))
        self.label_4.setStyleSheet(u"border: 2px solid red;")

        self.verticalLayout_2.addWidget(self.label_4)

        self.label_5 = QLabel(self.widget_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMaximumSize(QSize(582, 344))
        self.label_5.setStyleSheet(u"border: 2px solid red;")

        self.verticalLayout_2.addWidget(self.label_5)

        self.widget_3 = QWidget(self.widget_2)
        self.widget_3.setObjectName(u"widget_3")
        self.pushButton = QPushButton(self.widget_3)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(20, 20, 75, 24))
        self.label_6 = QLabel(self.widget_3)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(30, 70, 111, 31))
        self.label_7 = QLabel(self.widget_3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(40, 120, 221, 21))
        self.label_8 = QLabel(self.widget_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(40, 160, 221, 21))
        self.label_9 = QLabel(self.widget_3)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(40, 200, 221, 21))
        self.label_10 = QLabel(self.widget_3)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(280, 70, 121, 31))
        self.label_10.setStyleSheet(u"")
        self.textEdit = QTextEdit(self.widget_3)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(280, 110, 301, 131))
        self.pushButton_2 = QPushButton(self.widget_3)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(110, 20, 75, 24))

        self.verticalLayout_2.addWidget(self.widget_3)


        self.gridLayout.addWidget(self.widget_2, 0, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"left", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"top", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"right", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"front", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"rear", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"\u6f14\u793a\u6a21\u5f0f", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u76f8\u673a\u53cd\u9988:", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"left:", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"top:", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"right", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"result", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"\u76f8\u673a\u914d\u7f6e", None))
    # retranslateUi

