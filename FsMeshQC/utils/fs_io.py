import nibabel as nib
import numpy as np

def read_freesurfer_surf(path):
    """
    Read FreeSurfer .surf files (e.g., lh.white, rh.pial).
    Returns:
        V: (N,3) float64 vertex coordinates (mm, RAS)
        F: (M,3) int32   triangle face vertex indices (0-based)
    """
    V, F = nib.freesurfer.read_geometry(path)
    return V.astype(np.float64), F.astype(np.int32)
