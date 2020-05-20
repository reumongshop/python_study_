# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:38:14 2020

@author: USER
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        pixmap = QPixmap('web.png')
        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        lbl_size = QLabel('Width : ' + str(pixmap.width()) + ', Height : ' +str(pixmap.height()))
        lbl_size.setAlignment(Qt.AlignCenter)
        # 정렬! => 가로, 세로 정렬시키는데 가운데 정렬시키겠다!
        
        vbox = QVBoxLayout()
        vbox.addWidget(lbl_img)
        vbox.addWidget(lbl_size)
        self.setLayout(vbox)
        
        self.setWindowTitle('QPixmap')
        self.setGeometry(300, 300, 300, 200)
        self.show()
        
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())