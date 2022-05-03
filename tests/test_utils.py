import numpy as np

def same_labels(labels1, labels2):
    """
    Compare integer labels labels1 and labels2.
    If labels1 and labels2 have different structures, raises ValueError.

    labels1 = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
    ]
    and
    labels2 = [
        [0, 0, 0],
        [2, 2, 2],
        [1, 1, 1]
    ]
    are essentially the same therefore same_labels(labels1, labels2) does not raises error.
    Let labels3 = [
        [0, 0, 0],
        [1, 1, 2],
        [1, 1, 2]
    ]
    labels1 and labels2 have different structures. same_labels(labels1, labels3) raises error.
    """
    labels2 = labels2.astype(labels1.dtype)
    assert np.max(labels1) == np.max(labels2)
    for label in range(np.max(labels1)):
        indices_in_labels1 = np.where(labels1==label)
        label_in_labels2 = labels2[indices_in_labels1[0][0], indices_in_labels1[1][0]]
        if not ((labels1==label) == (labels2==label_in_labels2)).all():
            raise ValueError("labels1 and labels2 are not the same")
