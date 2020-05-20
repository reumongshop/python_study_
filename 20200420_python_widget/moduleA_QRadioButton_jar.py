# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:25:34 2020

@author: USER
"""

# =============================================================================
# 라디오 버튼은 일반적으로 사용자에게 여러 개 중 하나의 옵션을 선택하도록 할 때 사용

# 메소드
# text() : 버튼의 텍스트를 반환
# setText() : 라벨에 들어갈 텍스트 설정
# setChecked() : 버튼의 선택 여부 설정
# isChecked() : 버튼의 선택 여부 반환
# toggle() : 버튼의 상태 변경
# =============================================================================

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QRadioButton

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        rbtn1 = QRadioButton('First Button', self)
        rbtn1.move(50, 50)
        rbtn1.setChecked(True)
        
        rbtn2 = QRadioButton(self)
        rbtn2.move(50, 70)
        rbtn2.setText('Second Button')
        
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('QRadioButton')
        self.show()
        
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())