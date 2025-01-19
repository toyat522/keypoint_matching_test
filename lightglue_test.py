from lightglue.lightglue import LightGlue
from lightglue.superpoint import SuperPoint
from lightglue.utils import load_image, rbd
import lightglue.viz2d as viz2d

import matplotlib.pyplot as plt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image("./assets/image0.jpg", resize=[768, 1024]).to(device)
image1 = load_image("./assets/image1.jpg", resize=[768, 1024]).to(device)

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

# match the features
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01["matches"]  # indices with shape (K,2)
points0 = feats0["keypoints"][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1["keypoints"][matches[..., 1]]  # coordinates in image #1, shape (K,2)

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f"Stop after {matches01['stop']} layers")
plt.show()

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
plt.show()
