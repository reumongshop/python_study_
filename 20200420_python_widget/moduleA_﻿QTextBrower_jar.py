# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:17:40 2020

@author: USER
"""

# =============================================================================
# QTextBrower 클래스는
# 하이퍼텍스트 내비게이션을 포함한 리치 텍스트(서식있는 텍스트) 브라우저를 제공
# 이 클래스는 읽기 전용이며, QTextEdit 의 확장형으로,
# 하이퍼텍스트 문서의 링크들을 사용할 수 있음
# 편집 가능한 리치 텍스트 편집기를 사용하기 위해 QTextEdit을 사용해야함
# 또한, 하이퍼텍스트 네비게이션이 없는 텍스트 브라우저를 사용하기 위해
# QTextEdit을 setReadOnly()를 사용해서 편집이 불가능하도록 해줌
# 짧은 리치 텍스트를 표시하기 위해 QLabel을 사용할 수 있다!
# =============================================================================

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QTextBrowser, QPushButton, QVBoxLayout)

# import 할 때 한 줄로 쓸 경우는 상관없지만, 줄바꿈 될 시 괄호로 묶어야함!

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.le = QLineEdit()
        self.le.returnPressed.connect(self.append_text)
        # 키보드에 엔터를 눌렀는지 확인해주는 이벤트
        # 엔터를 누르면 append_text를 받아와서 입력값 자동 저장
        
        self.tb = QTextBrowser() #QTextBrowser() 클래스를 이용해서 텍스트 브라우저 객체 생성
        self.tb.setAcceptRichText(True) # 서식 있는 텍스트(Rich text)를 사용할 수 있음
                                        # 디폴트로 True이기 때문에 없어도 되는 부분! 
        self.tb.setOpenExternalLinks(True) # 외부 링크로의 연결 가능
        
        self.clear_btn = QPushButton('Clear') # 버튼을 클릭하면 
        self.clear_btn.pressed.connect(self.clear_text) # Clear_text 메소드 호출
        
        vbox = QVBoxLayout() # 레이아웃 해주기 위한 vbox 객체 생성!
        vbox.addWidget(self.le, 0) # .addWidget() 순서 정해줄 수 있음!
        vbox.addWidget(self.tb, 1)
        vbox.addWidget(self.clear_btn, 2)
        self.setLayout(vbox)
        self.setWindowTitle('QTextBrowser')
        self.setGeometry(300, 300, 300, 300)
        self.show()
        
    def append_text(self):
        text = self.le.text()
        self.tb.append(text)
        self.le.clear() # 입력값 초기화 시켜주는 함수!
        
    def clear_text(self):
        self.tb.clear()
        
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())