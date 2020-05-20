# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:32:52 2020

@author: USER
"""

# =============================================================================
# QComboBox
# 작은 공간을 차지하면서 여러 옵션들을 제공하고 그 중 하나의 옵션을 선택할 수 있도록 해주는 위젯
# =============================================================================

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.lbl = QLabel('Option1', self)
        self.lbl.move(50, 150)
        
        cb = QComboBox(self)
        cb.addItem('Option1')
        cb.addItem('Option2')
        cb.addItem('Option3')
        cb.addItem('Option4')
        cb.move(50, 50)
        
        cb.activated[str].connect(self.onActivated)
        # 현재 콤보 박스에 활성화 되어있는 글자를 뽑아오기!
        
        self.setWindowTitle('QComboBox')
        self.setGeometry(300, 300, 300, 200)
        self.show() 
        
    def onActivated(self, text):
        self.lbl.setText(text)
        self.lbl.adjustSize() # 전달된 글자에 맞춰 자동으로 라벨 사이즈 변경!
    
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())