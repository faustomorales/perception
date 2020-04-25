# pylint: disable=invalid-name

import os
import string

import pytest

from perception import hashers, testing

TEST_IMAGES = [
    os.path.join('tests', 'images', f'image{n}.jpg') for n in range(1, 11)
]


# The PDQ hash isometric computation is inexact. See
# https://github.com/faustomorales/pdqhash-python/blob/master/tests/test_compute.py
# for details.
@pytest.mark.parametrize(
    "hasher_class,pil_opencv_threshold,transform_threshold,opencv_hasher",
    [(hashers.AverageHash, 0.1, 0.1, False),
     (hashers.WaveletHash, 0.1, 0.1, False), (hashers.PHash, 0.1, 0.1, False),
     (hashers.PDQHash, 0.1, 0.15, False), (hashers.DHash, 0.1, 0.1, False),
     (hashers.MarrHildreth, 0.1, 0.1, True),
     (hashers.BlockMean, 0.1, 0.1, True),
     (hashers.ColorMoment, 10, 0.1, True)])
def test_image_hashing_common(hasher_class, pil_opencv_threshold,
                              transform_threshold, opencv_hasher):
    testing.test_image_hasher_integrity(
        hasher=hasher_class(),
        pil_opencv_threshold=pil_opencv_threshold,
        transform_threshold=transform_threshold,
        opencv_hasher=opencv_hasher)


def test_video_hashing_common():
    testing.test_video_hasher_integrity(
        hasher=hashers.FramewiseHasher(
            frame_hasher=hashers.PHash(hash_size=16),
            interframe_threshold=0.1,
            frames_per_second=1))


def test_video_reading():
    # We should get one red, one green, and one blue frame
    for frame, _, timestamp in hashers.tools.read_video(
            filepath='perception/testing/videos/rgb.m4v',
            frames_per_second=0.5):
        assert timestamp in [0.0, 2.0, 4.0]
        channel = int(timestamp / 2)
        assert frame[:, :, channel].min() > 230
        for other in [0, 1, 2]:
            if other == channel:
                continue
            assert frame[:, :, other].max() < 20


def test_common_framerate():
    assert hashers.tools.get_common_framerates(
        dict(zip(['a', 'b', 'c'], [1 / 3, 1 / 2, 1 / 5]))) == {
            1.0: ('a', 'b', 'c')
        }
    assert hashers.tools.get_common_framerates(
        dict(zip(['a', 'b', 'c'], [1 / 3, 1 / 6, 1 / 9]))) == {
            1 / 3: ('a', 'b', 'c')
        }
    assert hashers.tools.get_common_framerates(
        dict(
            zip(['a', 'b', 'c', 'd', 'e'],
                [1 / 3, 1 / 2, 1 / 5, 1 / 7, 1 / 11]))) == {
                    1.0: ('a', 'b', 'c', 'd', 'e')
                }
    assert hashers.tools.get_common_framerates(
        dict(zip(string.ascii_lowercase[:6],
                 [10, 5, 3, 1 / 3, 1 / 6, 1 / 9]))) == {
                     3.0: ('c', 'd', 'e', 'f'),
                     10.0: ('a', 'b')
                 }
    assert hashers.tools.get_common_framerates(
        dict(zip(['a', 'b'], [100, 1]))) == {
            100: ('a', 'b')
        }


def test_synchronized_hashing():
    video_hashers = {
        'phashframewise':
        hashers.FramewiseHasher(
            frame_hasher=hashers.PHash(hash_size=16),
            frames_per_second=1,
            interframe_threshold=0.2),
        'tmkl2':
        hashers.TMKL2(frames_per_second=15),
        'tmkl1':
        hashers.TMKL1(frames_per_second=15)
    }

    for filepath in [
            'perception/testing/videos/v1.m4v',
            'perception/testing/videos/v2.m4v'
    ]:
        # Ensure synchronized hashing
        hashes1 = {
            hasher_name: hasher.compute(filepath)
            for hasher_name, hasher in video_hashers.items()
        }
        hashes2 = hashers.tools.compute_synchronized_video_hashes(
            filepath=filepath, hashers=video_hashers)
        assert hashes1 == hashes2