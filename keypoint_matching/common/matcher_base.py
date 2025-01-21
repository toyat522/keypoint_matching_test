from abc import ABC, abstractmethod

class MatcherBase(ABC):
    @abstractmethod
    def load_image(self, img_path, resize, device=None):
        pass

    @abstractmethod
    def execute(self, img0_path, img1_path, max_num_keypoints, device=None):
        pass

    @abstractmethod
    def plot(self, img0, img1, m_kpts0, m_kpts1, kpts0, kpts1):
        pass
