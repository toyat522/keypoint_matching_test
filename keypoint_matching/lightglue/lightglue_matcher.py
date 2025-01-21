from ..common.matcher_base import MatcherBase
from .lightglue import LightGlue
from .viz2d import plot_images, plot_matches, plot_keypoints
from .superpoint import SuperPoint
from .utils import load_image, rbd

from typing import List, Tuple
import matplotlib.pyplot as plt
import torch

class LightGlueMatcher(MatcherBase):
    def load_image(self, img_path, resize, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(resize, List) or isinstance(resize, Tuple):
            resize = [resize[1], resize[0]]
        return load_image(img_path, resize=resize).to(device)

    def execute(self, img0, img1, max_num_keypoints, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # SuperPoint+LightGlue
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)

        # Extract local features
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)

        # Match the features
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        # Keep the matching keypoints
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return kpts0, kpts1, m_kpts0, m_kpts1

    def plot(self, img0, img1, m_kpts0, m_kpts1, kpts0, kpts1):
        plot_images([img0, img1])
        plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        plot_keypoints([kpts0, kpts1], colors=["blue", "blue"], ps=10)
        plt.show()

