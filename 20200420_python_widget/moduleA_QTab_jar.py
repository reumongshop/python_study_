# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:32:32 2020

@author: USER
"""


# =============================================================================
# QTabWidget
# 이러한 탭은 프로그램 안의 구성요소들이 많은 면적을 차지하지 않으면서,
# 그것들을 카테고리에 따라 분류할 수 있기 때문에 유용하게 사용될 수 있습니다.
# =============================================================================
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        tab1 = QWidget() # 슬라이더, 다이얼로 변경해도 됨!
        tab2 = QWidget() 
        
        tabs = QTabWidget()
        tabs.addTab(tab1, 'Tab1')
        tabs.addTab(tab2, 'Tab2')
        
        vbox = QVBoxLayout()
        vbox.addWidget(tabs)
        
        self.setLayout(vbox)
        
        self.setWindowTitle('QTabWidget')
        self.setGeometry(300, 300, 300, 200)
        self.show()
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())