# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:39:08 2020

@author: USER
"""

# =============================================================================
# QGridLayout 클래스는 격자 형태의 UI를 구성하는 데 사용
# 그리드 레이아웃을 생성하기 위해 QGridLayout 클래스 사용
# addWidget 메서드에서 행과 열의 인덱스를 차례로 입력 받
# =============================================================================

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QTextEdit)

class MyApp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        
        grid.addWidget(QLabel('Title: '), 0, 0) #저장할 라벨 객체 만들고 추가할 위젯 위치 : 0행 0열
        grid.addWidget(QLabel('Author: '), 1, 0)
        grid.addWidget(QLabel('Review: '), 2, 0)
        
        grid.addWidget(QLineEdit(), 0, 1) # QLineEdit() : 한줄
        grid.addWidget(QLineEdit(), 1, 1)
        grid.addWidget(QTextEdit(), 2, 1) # QTextEdit() : 여러줄
        
        self.setWindowTitle('QGridLayout')
        self.setGeometry(300, 300, 300, 200)
        self.show()
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())