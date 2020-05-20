# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:23:49 2020

@author: USER
"""

# =============================================================================
# QBoxLayout : 박스 레이아웃 클래스
# QHBoxLayout(horizon)
# QVBoxlayout(vertical)
# => 여러 위젯을 수평으로 정렬하는 레이아웃 클래스
# 수평, 수직의 박스를 하나 만드는데, 다른 레이아웃 박스를 넣을 수 있고 위젯을 배치할 수 있음
# 예제 코드에서 위젯의 가운데 아래 부분에 두 개의 버튼을 배치하기 위해 수평, 수직의 박스를 하나씩 사용
# =============================================================================

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        okButton = QPushButton('OK')
        cancelButton = QPushButton('Cancel')
        
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)
        hbox.addStretch(1)
        # 양 사이드(수평선)에 1:1 비율이 남도록 만들기
        
        vbox = QVBoxLayout()
        vbox.addStretch(3)
        vbox.addLayout(hbox)
        vbox.addStretch(1)
        # 수직선에 3:1 비율이 남도록 만들기
        
        self.setLayout(vbox)
        
        self.setWindowTitle('Box Layout')
        self.setGeometry(300, 300, 300, 300)
        self.show()
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
        
