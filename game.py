import numpy as np
from threading import Timer
from cv2.cv2 import countNonZero
from PIL import ImageGrab, Image
from time import sleep

from directkeys import PressKey, ReleaseKey, L, R


class Game:
    # 움직일 때 L: w key, 0: 가만히, R: d key
    # rect : 해당 사각형 만큼을 이미지 캡처해온다.
    def __init__(self, rect):
        self.rect = rect
        self.action_list = [L, 0, R]

        self.before_action = -1
        self.timer = Timer(0.01, self._key_daapress, [self.before_action])

    def init_state(self):
        self._get_state()

    def _key_press(self, action):
        ReleaseKey(self.action_list[action])

    # 이미지 캡쳐
    def _get_state(self):
        self.state = np.array(ImageGrab.grab(bbox=self.rect).resize((200, 200), Image.ANTIALIAS))

    # action이 가만히가 아니라면 움직임
    def _move(self, action):
        if action is not 1:
            PressKey(self.action_list[action])
            sleep(0.1)
            ReleaseKey(self.action_list[action])

    # 헥사곤 게임종료시 왼쪽 중간 사각형이 검은색인지 체크
    def _is_gameover(self):

        # 사각형에서 pixel의 평균을 구한 뒤 30 아래인 것만 구해 30 이하가 85% 이상이면 True
        height_middle = int(self.state.shape[0] / 2)
        array = np.mean(self.state[height_middle - 10: height_middle + 10, :20], axis=2)
        black = np.where(array < 30)
        shape = array.shape

        a = black[0].shape[0]
        return a / (shape[0] * shape[1]) >= 0.75

    # move based on action, if gameover - reward else + reward
    def step(self, action):
        self._move(action)
        stable_reward = 0.01 if action == 1 else 0

        done = self._is_gameover()
        if done:
            reward = -2
        else:
            reward = stable_reward + 0.01

        self._get_state()

        return reward, done


