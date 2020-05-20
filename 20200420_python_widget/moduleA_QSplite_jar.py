# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:13:45 2020

@author: USER
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QFrame, QSplitter
from PyQt5.QtCore import Qt

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        hbox = QHBoxLayout()
        
        # 프레임 설 : 위, 아래 통으로 하나 // 가운데는 왼쪽, 오른쪽 나누어짐
        # 상단에 배치할 프레임
        top = QFrame() 
        top.setFrameShape(QFrame.Box)
        
        # 가운데 왼쪽에 배치할 프레임
        midleft = QFrame()
        midleft.setFrameShape(QFrame.StyledPanel) 
        
        # 가운데 오른쪽에 배치할 프레임
        midright = QFrame()
        midright.setFrameShape(QFrame.Panel)
        
        # 아래에 배치할 프레임
        bottom = QFrame()
        bottom.setFrameShape(QFrame.WinPanel)
        bottom.setFrameShadow(QFrame.Sunken)
        
        # 쪼개지는 방향 정함
        splitter1 = QSplitter(Qt.Horizontal) # 가로로 쪼개기!
        splitter1.addWidget(midleft)
        splitter1.addWidget(midright)
        
        splitter2 = QSplitter(Qt.Vertical)
        splitter2.addWidget(top)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(bottom)
        
        hbox.addWidget(splitter2)
        self.setLayout(hbox)
        
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('QSplitter')
        self.show()
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())