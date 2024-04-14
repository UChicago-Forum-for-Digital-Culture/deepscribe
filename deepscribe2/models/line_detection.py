# models for extracting sign order from hotspot positions
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.linear_model import RANSACRegressor, Ridge
import matplotlib.pyplot as plt


def dist_to_line_2d(slope, intercept, point):
    return np.abs(-slope * point[0] + point[1] - intercept) / np.sqrt((slope**2 + 1))


# TODO: this needs some work to really behave like a clusterer
# although is that even what it is? Needs to group points into lines
# AND sort the lines... maybe that's best done separately?
class SequentialRANSAC(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        max_lines=20,
        residual_threshold=50,
        max_trials=5000,
        max_slope=0.15,
        ridge_alpha=500,
        assign_remaining_to_closest=True,
    ):
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.ridge_alpha = ridge_alpha
        self.max_slope = max_slope
        self.max_lines = max_lines
        self.assign_remaining_to_closest = assign_remaining_to_closest

    def fit(self, X, y=None):
        self.labels_ = np.full((X.shape[0]), -1)
        fit_mask = np.full((X.shape[0]), True)
        inds = np.arange(0, X.shape[0])
        current_label = 0
        iters = 0
        self.ransac_lines = []

        # keep fitting RANSAC estimators  until there are insufficient points to fit.
        while (
            np.any(self.labels_ == -1)
            and iters < self.max_lines
            and np.sum(self.labels_ == -1) > 2
        ):
            ransac_line = RANSACRegressor(
                estimator=Ridge(alpha=self.ridge_alpha),
                residual_threshold=self.residual_threshold,
                is_model_valid=lambda m, _x, _y: np.abs(m.coef_) < self.max_slope,
                max_trials=self.max_trials,
                min_samples=2,
            )
            # this usually happens when RANSAC can't find
            # a valid consensus sample. If this happens, just
            # stop fitting and assign the rest of the data to the closest line.
            try:
                ransac_line.fit(X[fit_mask, 0].reshape(-1, 1), X[fit_mask, 1])
            except ValueError as e:
                # if concensus set
                print(e)
                break

            self.ransac_lines.append(ransac_line)

            # use inlier mask and fit mask to index into true indices

            true_inds = inds[fit_mask][ransac_line.inlier_mask_]

            self.labels_[true_inds] = current_label
            fit_mask[true_inds] = False
            current_label += 1
            iters += 1

        # POSTPROCESSING: assign any unassigned to closest line
        if self.assign_remaining_to_closest:
            (unassigned,) = np.where(self.labels_ == -1)
            # TODO: refactor and test this
            for unassigned_pt in unassigned:
                dists = [
                    dist_to_line_2d(
                        fitted_line.estimator_.coef_[0],
                        fitted_line.estimator_.intercept_,
                        X[unassigned_pt, :],
                    )
                    for fitted_line in self.ransac_lines
                ]
                self.labels_[unassigned_pt] = np.argmin(dists)

        # sort the lines and points
        # rough heuristic: sort the lines by y-intercept, then sort points by x_coordinate
        line_intercepts = [
            fitted_line.estimator_.intercept_ for fitted_line in self.ransac_lines
        ]
        # not inverted - top-left corner is 0, 0
        # TODO: fix behavior when no longer assigning remaining to closest
        label_ordering = np.argsort(line_intercepts)
        ordered_pts = []
        for label in label_ordering:
            (point_inds,) = np.where(self.labels_ == label)
            label_points_sorted = np.argsort(X[point_inds, 0])  # sort by x coordinate
            ordered_pts.append(point_inds[label_points_sorted])
        self.ordering_ = np.hstack(ordered_pts)

        return self


def plot_centroids(centroids, labels, ax, ordering=None):
    for lab in np.unique(labels):
        ax.scatter(
            centroids[labels == lab, 0],
            centroids[labels == lab, 1],
            label=f"Line {lab}",
        )

    if ordering is not None:
        for i, pt_idx in enumerate(ordering):
            ax.annotate(str(i), (centroids[pt_idx, 0], centroids[pt_idx, 1]))
