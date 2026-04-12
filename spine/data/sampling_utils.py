import numpy as np
from scipy.spatial import KDTree

from .distribution_utils import get_distribution


class PatchSampler:
    def __init__(self, distribution: str='batch_128', min_samples=2):
        self.distribution = distribution
        self.distribution_func = get_distribution(distribution)
        self.min_samples = min_samples

    def sample_nearest_patch(self, coords, num_samples):
        num_samples = min(len(coords), num_samples)

        if num_samples == len(coords):
            return np.arange(len(coords))

        # Build a KDTree for efficient nearest neighbor searches
        tree = KDTree(coords)
        
        # Randomly choose an index as the starting point for the patch
        center_idx = np.random.randint(0, len(coords))
        center_coord = coords[center_idx]
        
        # Query the nearest 'num_samples' points including the center itself
        _, idx_nearest = tree.query(center_coord, k=num_samples)
        
        # Fetch the coordinates of these nearest points
        return idx_nearest

    def get_distribution_expectation(self):
        return np.mean([self.distribution() for _ in range(10000)])

    def __call__(self, coords):
        total_samples = max(self.min_samples, int(len(coords) * self.distribution_func()))
        return self.sample_nearest_patch(coords, total_samples)


if __name__ == "__main__":
    # Test the patch sampler
    coords = np.random.rand(100, 2)
    sampler = PatchSampler("beta_3_1")
    # print(sampler(coords).shape)
    print(sampler.get_distribution_expectation())