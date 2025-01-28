from keypoint_matching.common.keypoint_matcher import KeypointMatcher
from keypoint_matching.lightglue.lightglue_matcher import LightGlueMatcher

kpt_matcher = KeypointMatcher()
kpt_matcher.set_matcher(LightGlueMatcher(
    max_num_keypoints=1024,
    depth_confidence=0.9,
    width_confidence=0.95,
))

print("LightGlue model:")
print(kpt_matcher.get_matcher_model())
print()

print("SuperPoint model:")
print(kpt_matcher.get_descriptor_model())
print()

img0 = kpt_matcher.load_image("./assets/image0.jpg")
img1 = kpt_matcher.load_image("./assets/image1.jpg")
kpts0, kpts1, m_kpts0, m_kpts1 = kpt_matcher.execute(img0, img1)

print("Extraction time:")
print(kpt_matcher.get_extraction_time())
print()

print("Matching time:")
print(kpt_matcher.get_matching_time())
print()

kpt_matcher.plot(img0, img1, m_kpts0, m_kpts1, kpts0, kpts1)

