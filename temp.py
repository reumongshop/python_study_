# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# PyQt5 : Python GUI 기본 모듈
# QApplication 클래스 : 윈도우 기본, GUI 응용프로그램의 제어 흐름과 기본 설정들을 관리
# QWidget : 꾸미기, 아이콘
# 필요한 모듈 불러오기
# 기본적인 UI 구성요소를 제공하는 위젯(클래스)들은 PyQt5.QtWidgets 모듈에 포함되어 있음
# QtWidgets 모듈에 포함된 모든 클래스들과 이에 대한 자세한 설명은 QtWidgets 공식 문서에서 확인 가능
# =============================================================================
# self.setwindowTitle('My First Application')
# self.move(300, 300)
# self.resize(400, 200)
# self.show()
# =============================================================================
# self : MyApp 객체
# self.setWindowTitle() 메소드 : 타이틀바에 나타나는 창의 제목 설정
# move() 메소드 : 위젯을 스크린의 x=300px, y=300px 위치로 이동
# resize() 메소드 : 위젯의 크기를 너비 400px, 높이 200px 로 조절
# show() 메소드 : 윈도우에 위젯을 보여준다
# =============================================================================

import sys
from PyQt5.QtWidgets import QApplication, QWidget

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()
        
    if __name__ == '__main__': #본인이 직접 실행
        app = QApplication(sys.argv)
        ex = MyApp()
        sys.exit(app.exec_())
        
         
        
        
 