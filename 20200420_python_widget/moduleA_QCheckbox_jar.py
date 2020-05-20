# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:04:47 2020

@author: USER
"""

# =============================================================================
# QCheckBox
# 위젯은 on(체크됨) / off(체크안됨)의 두 상태를 갖는 버튼을 제공
# 이 위젯은 하나의 텍스트 라벨과 함께 체크 박스를 제공
# 체크 박스가 선택되너가 해제될 때, stateChanged() 시그널을 발생
# 체크 박스의 상태가 변할 때마다 어떠한 동작을 발생시키고 싶을 때,
# 이 시그널을 특정 슬롯에 연결할 수 있음
# 또한 체크 박스의 선택 여부를 확인하기 위해서,
# isChecked() 메소드를 사용할 수 있음
# 선택 여부에 따라 boolean 값을 반환

# 일반적인 체크박스는 선택/해제 상태만을 갖지만,
# setTristate() 메소드를 사용하면 '변경없음(no change)' 상태를 가질 수 있음
# 이 체크 박스는 사용자에게 선택하거나 선택하지 않을 옵션을 줄 때 유용

# 세가지 상태를 갖는 체크박스의 상태를 얻기 위해서는
# checkState() 메소드를 사용
# 선택/변경없음/해제 여부에 따라 각각 2/1/0 값을 반환

# QCheck 위젯이 갖고 있는 view 함수
# text() : 체크 박스의 라벨 텍스트 반환
# setText() : 체크 박스의 라벨 텍스트 설정
# isChecked() : 체크 박스의 상태 반환(True/False)
# checkState() : 체크 박스의 상태 반환(2/1/0)
# toggle() : 체크 박스의 상태를 변경

# 위젯에서는 시그널이란 말을 더 많이 사용
# 이벤트라고 생각하면 됨!

# 시그널
# pressed() : 체크 박스를 누를 때 신호 발생
# released() : 체크 박스에서 뗄 때 신호 발생
# clicked() : 체크 박스를 클릭할 때 신호 발생
# stateChanged() : 체크 박스의 상태가 바뀔 때 신호 발생
# =============================================================================

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox
from PyQt5.QtCore import Qt

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        cb = QCheckBox('Show Title', self)
        cb.move(20, 20)
        cb.toggle()
        cb.stateChanged.connect(self.changeTitle)
        # connect 함수 : 이벤트가 발생했을시 해당 함수 호출
        # 상태 변화가 일어나야함
        
        self.setWindowTitle('QCheckBox')
        self.setGeometry(300, 300, 300, 200)
        self.show()
        
    def changeTitle(self, state):
        if state == Qt.Checked:
            self.setWindowTitle('QCheckBox') # 'on' 라고 써도 됨!
        else:
            self.setWindowTitle('') # 'off' 라고 써도 됨!
            
            
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())