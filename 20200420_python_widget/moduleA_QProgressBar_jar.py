# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:05:51 2020

@author: USER
"""

# =============================================================================
# QProgressBar 위젯은 수평, 수직의 진행 표시줄을 제공
# setMinimum()과 setMaximum() 메소드로 진행 표시줄의 최소값과 최대값을 설정할 수 있으며,
# 또는 setRange() 메소드로 한 번에 범위를 설정할 수도 있다
# 기본값은 0과 99
# setValue() 메소드로 진행 표시줄의 진행 상태를 특정 값으로 설정할 수 있고,
# reset() 메소드는 초기 상태로 되돌린다

# 진행 표시줄의 최소값과 최대값을 모두 0으로 설정하면,
# 진행 표시줄은 위의 그림과 같이 항상 진행 중인 상태로 표시됨
# 이 기능은 다운로드하고 있는 파일의 용량을 알 수 없을 때 유용하게 사용 가능
# =============================================================================


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar
from PyQt5.QtCore import QBasicTimer

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)
        
        self.btn = QPushButton('Start', self)
        self.btn.move(40, 80)
        self.btn.clicked.connect(self.doAction)
        # 이벤트가 실행된 함수는 아래에 선언할 것!
        
        self.timer = QBasicTimer()
        self.step = 0 #기본값은 0 / 1씩 증가시키기!
        
        self.setWindowTitle('QProgressBar')
        self.setGeometry(300, 300, 300, 200)
        self.show()
    
    def timerEvent(self, e): # 시간의 흐름에 따라 이벤트 발생 / 상속받은 함수 // 재정의 함수!
        if self.step >= 100:
            self.timer.stop()
            self.btn.setText('Finished')
            return
        
        self.step = self.step +1
        self.pbar.setValue(self.step)
        
    def doAction(self):
        if self.timer.isActive(): # 타이머가 활동 중인지 알아내는 함수
            self.timer.stop()
            self.btn.setText('Start') # 첫번째 값이 종료값!
            
        else:
            self.timer.start(100, self)
            self.btn.setText('Stop')
            

if __name__ == '__main__': #본인이 직접 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
        