import numpy as np
from cv2.cv2 import countNonZero
from PIL import ImageGrab

from directkeys import PressKey, L, R


class Game:
    # 움직일 때 L: w key, 0: 가만히, R: d key
    # rect : 해당 사각형 만큼을 이미지 캡처해온다.
    def __init__(self, rect):
        self.rect = rect

        self.current_reward = 0

        self.action_list = [L, 0, R]
        self.current_action = 1

    # 이미지 캡쳐
    def _get_state(self):
        self.state = np.array(ImageGrab.grab(bbox=self.rect))

    # action이 가만히가 아니라면 움직임
    def _move(self, action):
        if action is not 1:
            PressKey(self.action_list[action])

    # 헥사곤 게임종료시 왼쪽 중간 사각형이 검은색인지 체크
    def _is_gameover(self):
        def _rgb2gray(rgb) -> float:
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray

        # change rgb -> gray, get middle of image, get rect from middle
        gray = _rgb2gray(self.state)
        height_middle = int(gray.shape[0] / 2)
        is_black = countNonZero(gray[height_middle - 15: height_middle + 15, :30]) == 0
        return is_black
