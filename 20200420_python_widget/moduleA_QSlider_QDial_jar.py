# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:38:46 2020

@author: USER
"""

# =============================================================================
# QSlider : 수평 또는 수직 방향의 슬라이더
# 생성자에서 가로 세로 배치 형상을
# Qt.Orientation 상수(Qt.Horizontal 또는 Qt.Vertical)에 담아 생성할 수 있다

# setTickInterval() 메소드 : 슬라이더의 틱(tick)의 간격을 조절하기 위해서 사용하는 메소드
# 메소드의 입력값은 픽셀이 아니라 값을 의미

# setTickPosition() 메소드 : 틱의 위치를 조절하기 위해서 사용하는 메소드

# QSlider.NoTicks 0 : 틱을 표시하지 않음
# QSlider.TicksAbove 1 : 틱을 (수평) 슬라이더 위쪽에 표시
# QSlider.TicksBelow 2 : 틱을 (수평) 슬라이더 아래쪽에 표시
# QSlider.TicksBothSides 3: 틱을 (수평) 슬라이더 양쪽에 표시
# QSlider.TicksLeft TicksAbove : 틱을 (수직) 슬라이더 왼쪽에 표시
# QSlider.TicksRight TicksBelow : 틱을 (수직) 슬라이더 오른쪽에 표시

# QDial 위젯
# setNotchesVisible() 메소드 : 다이얼 위젯에 노치(notch)를 표시하기 위해서 사용
# True로 설정하면 둥근 다이얼을 따라서 노치들이 표시
# 기본적으로 노치는 표시되지 않도록 설정되어 있음

# QSlider과 QDial 위젯에서 가장 자주 쓰이는 시그널 // 이벤트!
# valueChanged() : 슬라이더의 값이 변할 때 발생
# sliderPressed() : 사용자가 슬라이더를 움직이기 시작할 때 발생
# sliderMoved() : 사용자가 슬라이더를 움직이면 발생
# sliderReleased() : 사용자가 슬라이더를 놓을 때 발생

# 예제에서는 valueChanged 시그널 사용
# =============================================================================

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QDial, QPushButton
from PyQt5.QtCore import Qt

# 슬라이더, 다이얼, 버튼 이벤트 3개!
class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.move(30, 30) # 슬라이더 위치는 x축 30, y축 30 떨어진 곳에
        self.slider.setRange(0, 50) # 슬라이더 범위
        self.slider.setSingleStep(2) # 슬라이더는 하나로만 잡고(SingleStep) 2씩 증가!
        
        self.dial = QDial(self)
        self.dial.move(30, 50) # 슬라이더 보다 아래쪽에 위치 
        self.dial.setRange(0, 50) # 슬라이더와 동일한 범위

        btn = QPushButton('Default', self)
        btn.move(35, 160) 
        
        self.slider.valueChanged.connect(self.dial.setValue)
        # 슬라이더 쪽에 변화가 일어난 후, 다이얼값이 슬라이더와 동일하게 발생할 수 있도록!
        # => 슬라이더가 움직이면 다이얼도 움직이도록 하는 것!
        # valueChanged() : 슬라이더 값이 변할 때 발생!
        self.dial.valueChanged.connect(self.slider.setValue)
        btn.clicked.connect(self.button_clicked)
        
        self.setWindowTitle('QSlider and QDial')
        self.setGeometry(300, 300, 400, 200)
        self.show()
        
    def button_clicked(self):
        self.slider.setValue(0)
        self.dial.setValue(0) #버튼 누르면 초기화 시키려고 값을 0으로 줌!
        
if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
        
        