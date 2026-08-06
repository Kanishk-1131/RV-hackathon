"""
AuraGuard detector: camera physical position / orientation drift.

Detects when a camera has SETTLED into a new pan/tilt orientation, as
distinct from transient shake, a brief bump, or subject motion in an
otherwise-static scene.

Signal contract:
    {"signal_name", "triggered", "confidence", "evidence", "bbox"}
"""

from collections import deque

import cv2
import numpy as np

SIGNAL_NAME = "camera_position_drift"


class CameraPositionDetector:
    """Detects sustained pan/tilt displacement from the calibration baseline.

    Reads the reference frame from a shared AdaptiveBaseline-like object
    rather than running its own calibration. Requires only two attributes
    on that object: ``ready`` (bool) and ``reference_frame`` (BGR ndarray).

    Degrees are derived from the pinhole model: the image centre is mapped
    through the estimated homography, and the resulting pixel displacement
    is converted to an angle via the focal length implied by ``hfov_deg``.

    Args:
        baseline: AdaptiveBaseline-like object (``ready``, ``reference_frame``).
        hfov_deg: Camera horizontal field of view in degrees. This is the
            single most important calibration constant -- see module notes.
        pan_threshold_deg: Horizontal displacement to consider significant.
        tilt_threshold_deg: Vertical displacement to consider significant.
        sustain_frames: Consecutive qualifying frames required to trigger.
        stability_tol_deg: Max std-dev across the window for the estimate to
            count as "settled" rather than "shaking".
        min_matches: Below this many good matches, report low confidence
            instead of guessing.
        min_inlier_ratio: Below this RANSAC inlier ratio, the homography is
            considered untrustworthy.
        n_features: ORB feature budget per frame.
    """

    def __init__(
        self,
        baseline=None,
        hfov_deg=90.0,
        pan_threshold_deg=2.0,
        tilt_threshold_deg=2.0,
        sustain_frames=20,
        stability_tol_deg=0.75,
        min_matches=12,
        min_inlier_ratio=0.35,
        n_features=1000,
    ):
        self.baseline = baseline
        self.hfov_deg = float(hfov_deg)
        self.pan_threshold_deg = float(pan_threshold_deg)
        self.tilt_threshold_deg = float(tilt_threshold_deg)
        self.sustain_frames = int(sustain_frames)
        self.stability_tol_deg = float(stability_tol_deg)
        self.min_matches = int(min_matches)
        self.min_inlier_ratio = float(min_inlier_ratio)

        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._ref_kp = None
        self._ref_des = None
        self._ref_shape = None
        self._ref_unusable = None
        self._history = deque(maxlen=self.sustain_frames)
        self._streak = 0

    # ---------------------------------------------------------------- utils

    def reset(self):
        """Clear internal state, including the cached reference keypoints."""
        self._ref_kp = None
        self._ref_des = None
        self._ref_shape = None
        self._ref_unusable = None
        self._history.clear()
        self._streak = 0

    @staticmethod
    def _signal(triggered, confidence, evidence, bbox=None):
        return {
            "signal_name": SIGNAL_NAME,
            "triggered": bool(triggered),
            "confidence": float(round(confidence, 3)),
            "evidence": evidence,
            "bbox": bbox,
        }

    def _focal_px(self, width):
        """Focal length in pixels implied by the configured horizontal FOV."""
        return (width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)

    def _ensure_reference(self):
        """Lazily extract reference keypoints once calibration completes."""
        if self._ref_des is not None:
            return True
        if self.baseline is None or not getattr(self.baseline, "ready", False):
            return False
        ref = getattr(self.baseline, "reference_frame", None)
        if ref is None:
            return False

        gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.min_matches:
            # Calibration finished, but the reference view is too featureless
            # to anchor position tracking against.
            self._ref_unusable = len(kp) if kp is not None else 0
            return False

        self._ref_kp = kp
        self._ref_des = des
        self._ref_shape = ref.shape[:2]
        return True

    def _match(self, des):
        """Lowe-ratio filtered matches against the reference descriptors."""
        if des is None or len(des) < 2:
            return []
        pairs = self._matcher.knnMatch(des, self._ref_des, k=2)
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    # ----------------------------------------------------------- core check

    def check(self, frame):
        """Analyse one BGR frame and return a signal dict."""
        if frame is None or frame.size == 0:
            return self._signal(False, 0.0, "empty frame received; no analysis performed")

        if not self._ensure_reference():
            unusable = getattr(self, "_ref_unusable", None)
            if unusable is not None:
                return self._signal(
                    False, 0.1,
                    f"baseline reference frame yielded only {unusable} trackable "
                    f"features (minimum {self.min_matches}); scene is too featureless "
                    f"to anchor position tracking -- camera may be aimed at a blank "
                    f"surface or already obstructed at calibration time",
                )
            return self._signal(
                False, 0.0,
                "baseline calibration not complete; awaiting reference frame before "
                "position tracking can begin",
            )

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)

        good = self._match(des)
        if len(good) < self.min_matches:
            self._streak = 0
            self._history.clear()
            return self._signal(
                False, 0.15,
                f"only {len(good)} feature matches against baseline (minimum "
                f"{self.min_matches}); scene may be obstructed, blank, or severely "
                f"degraded -- position cannot be assessed",
            )

        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self._ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask is None:
            self._streak = 0
            self._history.clear()
            return self._signal(
                False, 0.15,
                f"homography estimation failed across {len(good)} matches; geometry "
                f"between current view and baseline is inconsistent",
            )

        inliers = int(mask.sum())
        inlier_ratio = inliers / max(len(good), 1)
        if inlier_ratio < self.min_inlier_ratio:
            self._streak = 0
            self._history.clear()
            return self._signal(
                False, 0.2,
                f"only {inliers}/{len(good)} matches ({inlier_ratio:.0%}) fit a "
                f"consistent transform; correspondence too unreliable to judge "
                f"camera position",
            )

        # Map the image centre through the inverse transform to get displacement
        # of the *current* view relative to the baseline view.
        fx = fy = self._focal_px(w)
        centre = np.float32([[[w / 2.0, h / 2.0]]])
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self._streak = 0
            return self._signal(False, 0.15, "transform matrix was singular; frame skipped")
        moved = cv2.perspectiveTransform(centre, H_inv)[0][0]
        # Displacement of SCENE CONTENT between baseline view and current view.
        dx = float(moved[0] - w / 2.0)
        dy = float(moved[1] - h / 2.0)

        # Camera aim moves OPPOSITE to scene content: pan the camera right and
        # the scene slides left. Convention: +pan = aim right, +tilt = aim down.
        pan_deg = float(np.degrees(np.arctan2(-dx, fx)))
        tilt_deg = float(np.degrees(np.arctan2(-dy, fy)))
        roll_deg = float(np.degrees(np.arctan2(H_inv[1, 0], H_inv[0, 0])))

        self._history.append((pan_deg, tilt_deg))

        exceeds = (
            abs(pan_deg) > self.pan_threshold_deg
            or abs(tilt_deg) > self.tilt_threshold_deg
        )

        # Stability: the estimate must AGREE with itself across the window.
        # Large magnitude + high variance = shake, not a settled reposition.
        stable = False
        pan_sd = tilt_sd = float("nan")
        if len(self._history) == self._history.maxlen:
            arr = np.array(self._history)
            pan_sd = float(arr[:, 0].std())
            tilt_sd = float(arr[:, 1].std())
            stable = (
                pan_sd < self.stability_tol_deg and tilt_sd < self.stability_tol_deg
            )

        if exceeds:
            self._streak += 1
        else:
            self._streak = 0

        geom_conf = min(1.0, inlier_ratio * 1.2)

        if self._streak >= self.sustain_frames and stable:
            magnitude = max(
                abs(pan_deg) / self.pan_threshold_deg,
                abs(tilt_deg) / self.tilt_threshold_deg,
            )
            confidence = min(0.99, 0.55 + 0.25 * geom_conf + 0.1 * min(magnitude, 2.0))
            evidence = (
                f"camera settled at {abs(pan_deg):.1f} deg "
                f"{'right' if pan_deg > 0 else 'left'} pan (length) and "
                f"{abs(tilt_deg):.1f} deg {'down' if tilt_deg > 0 else 'up'} tilt "
                f"(height) from baseline orientation, sustained for {self._streak} "
                f"consecutive frames; estimate stable to within "
                f"{max(pan_sd, tilt_sd):.2f} deg across the window, "
                f"{inliers}/{len(good)} feature matches consistent"
            )
            if abs(roll_deg) > 1.0:
                evidence += f", roll {roll_deg:+.1f} deg"
            return self._signal(True, confidence, evidence)

        if exceeds and not stable:
            return self._signal(
                False, 0.3,
                f"displacement of {pan_deg:+.1f} deg pan / {tilt_deg:+.1f} deg tilt "
                f"detected but still fluctuating (spread {max(pan_sd, tilt_sd):.2f} deg "
                f"exceeds {self.stability_tol_deg} deg tolerance); consistent with "
                f"vibration or a transient bump rather than a settled reposition",
            )

        if exceeds:
            return self._signal(
                False, 0.3,
                f"displacement of {pan_deg:+.1f} deg pan / {tilt_deg:+.1f} deg tilt "
                f"observed for {self._streak}/{self.sustain_frames} frames; "
                f"not yet sustained long enough to confirm",
            )

        return self._signal(
            False, min(0.9, geom_conf),
            f"camera orientation within tolerance of baseline "
            f"({pan_deg:+.1f} deg pan, {tilt_deg:+.1f} deg tilt); "
            f"{inliers}/{len(good)} matches consistent",
        )
