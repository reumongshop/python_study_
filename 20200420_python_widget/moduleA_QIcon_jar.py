# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# =============================================================================
# 어플리케이션 아이콘은 타이틀바의 왼쪽 끝에 보여질 작은 이미지입니다. 어플리케이션 아이콘을 표시하는 방법
# 우선 폴더 안에, 아래와 같이 아이콘으로 사용할 이미지 파일을 저장
# =============================================================================

import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('web.png')) # 아이콘 심어주는 함수
        self.setGeometry(300, 300, 300, 200)
        self.show()
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
        