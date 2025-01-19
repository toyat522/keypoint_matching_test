from superglue.matching import Matching
from superglue.utils import make_matching_plot, read_image

import matplotlib.cm as cm
import torch

# Load the SuperPoint and SuperGlue models.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running inference on device \"{}\"".format(device))
config = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 1024 
    },
    "superglue": {
        "weights": "indoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2,
    }
}
matching = Matching(config).eval().to(device)

name0, name1 = ["./assets/image0.jpg", "./assets/image1.jpg"]
viz_path = "./assets/superglue_match.jpg"

# Load the image pair.
image0, inp0, scales0 = read_image(
    name0, device, resize=[1024, 768], rotation=0, resize_float=False)
image1, inp1, scales1 = read_image(
    name1, device, resize=[1024, 768], rotation=0, resize_float=False)

# Perform the matching.
pred = matching({"image0": inp0, "image1": inp1})
pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
matches, conf = pred["matches0"], pred["matching_scores0"]

# Write the matches to disk.
out_matches = {"keypoints0": kpts0, "keypoints1": kpts1,
                "matches": matches, "match_confidence": conf}

# Keep the matching keypoints.
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

# Visualize the matches.
color = cm.jet(mconf)
text = [
    "SuperGlue",
    "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
    "Matches: {}".format(len(mkpts0)),
]

make_matching_plot(
    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
    text, viz_path, show_keypoints=True)
