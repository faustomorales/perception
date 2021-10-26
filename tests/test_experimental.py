# pylint: disable=protected-access,invalid-name
import os
import tempfile
import sqlite3
import imgaug
import cv2
import numpy as np
import pandas as pd
import perception.hashers as ph
import perception.testing as pt
import perception.benchmarking as pb
import perception.hashers.tools as pht
import perception.benchmarking.image_transforms as pbit
import perception.experimental.local_descriptor_deduplication as ldd
import perception.experimental.approximate_deduplication as ad
import perception.experimental.ann as pea


def test_sift_deduplication():
    tdir = tempfile.TemporaryDirectory()
    watermark = cv2.cvtColor(
        cv2.imread(pt.DEFAULT_TEST_LOGOS[0], cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGRA2RGBA)
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, 'test')
               for filepath in pt.DEFAULT_TEST_IMAGES]).transform(
                   transforms={
                       'noop':
                       lambda image: image,
                       'pad':
                       imgaug.augmenters.Pad(percent=0.1),
                       'crop':
                       imgaug.augmenters.Crop(percent=0.1),
                       'watermark':
                       pbit.apply_watermark(watermark, alpha=1, size=0.8)
                   },
                   storage_dir=tdir.name)
    df = transformed._df.set_index('filepath')
    pairs = ldd.deduplicate(filepaths=df.index)
    clustered = pd.DataFrame(ad.pairs_to_clusters(
        ids=df.index, pairs=pairs)).set_index('id').merge(
            df, left_index=True, right_index=True).reset_index()
    n_clusters = clustered['cluster'].nunique()
    n_transforms = clustered['transform_name'].nunique()
    perfect = clustered.groupby('cluster').apply(
        lambda g: g['guid'].nunique() == 1 and g['transform_name'].nunique() == n_transforms
    ).sum()
    tainted = clustered.groupby('cluster')['guid'].nunique().gt(1).sum()
    pct_perfect = perfect / n_clusters
    pct_tainted = tainted / n_clusters
    assert pct_perfect > 0.1
    assert pct_tainted == 0


def test_validation_for_overlapping_case():
    tdir = tempfile.TemporaryDirectory()
    # Each image will have the center of the other
    # pasted in the top left corner.
    image1 = pht.read(pt.DEFAULT_TEST_IMAGES[0])
    image2 = pht.read(pt.DEFAULT_TEST_IMAGES[1])
    image1[:100, :100] = image2[100:200, 100:200]
    image2[:100, :100] = image1[100:200, 100:200]
    fp1 = os.path.join(tdir.name, 'test1.jpg')
    fp2 = os.path.join(tdir.name, 'test2.jpg')
    cv2.imwrite(fp1, image1[..., ::-1])
    cv2.imwrite(fp2, image2[..., ::-1])
    kp1, des1, dims1 = ldd.generate_image_descriptors(fp1)
    kp2, des2, dims2 = ldd.generate_image_descriptors(fp2)
    # These images should not match.
    assert not ldd.validate_match(
        kp1=kp1, kp2=kp2, des1=des1, des2=des2, dims1=dims1, dims2=dims2)


def test_handling_bad_file_case(caplog):
    tdir = tempfile.TemporaryDirectory()
    missing_file = os.path.join(tdir.name, 'missing-file')
    bad_file_handle = tempfile.NamedTemporaryFile()
    bad_file = bad_file_handle.name
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, 'test')
               for filepath in pt.DEFAULT_TEST_IMAGES]).transform(
                   transforms={
                       'noop': lambda image: image,
                   },
                   storage_dir=tdir.name)
    df = transformed._df.set_index('filepath')
    df.loc[missing_file] = df.iloc[0]
    df.loc[bad_file] = df.iloc[0]
    pairs = ldd.deduplicate(filepaths=df.index)
    clustered = pd.DataFrame(ad.pairs_to_clusters(
        ids=df.index, pairs=pairs)).set_index('id').merge(
            df, left_index=True, right_index=True).reset_index()

    assert bad_file not in clustered.index
    assert missing_file not in clustered.index

    bad_file_error = next(
        record for record in caplog.records if bad_file in record.message)
    assert bad_file_error
    assert bad_file_error.levelname == "ERROR"

    missing_file_warning = next(
        record for record in caplog.records if missing_file in record.message)
    assert missing_file_warning
    assert missing_file_warning.levelname == "WARNING"


def test_approximate_nearest_neighbors():
    for hasher, threshold in [(ph.PHash(), 0.1),
                              (ph.PHashU8(exclude_first_term=True), 5)]:
        metadata = pd.DataFrame(
            hasher.compute_parallel(pt.DEFAULT_TEST_IMAGES))
        metadata = pd.concat([metadata for n in range(100)], ignore_index=True)

        # Create a temporary database of hashes.
        table = "test-hashes"
        con = sqlite3.connect("")
        with con:
            metadata.assign(
                id=np.arange(len(metadata)),
                hash=metadata["hash"].apply(
                    lambda h: hasher.string_to_vector(h).tobytes())).to_sql(
                        name=table, con=con)

        # Create the nearest neighbors object from the database.
        ann = pea.ApproximateNearestNeighbors.from_database(
            con=con,
            table=table,
            paramstyle="qmark",
            hash_length=hasher.hash_length,
            metadata_columns=["filepath", "hash"],
            dtype=hasher.dtype,
            distance_metric=hasher.distance_metric)

        # Make the search exhaustive.
        ann.set_nprobe(ann.nlist)

        # Create a non-exact query.
        query_filepath = metadata.iloc[3]["filepath"]
        query_hash = hasher.compute(cv2.blur(pht.read(query_filepath), (3, 3)))

        # Obtain matches.
        matches = pd.json_normalize(
            ann.search(
                queries=[{
                    "id": "test-id",
                    "hash": query_hash
                }],
                threshold=threshold,
                k=100)[0]["matches"])

        # Ensure that we (1) found matches, (2) that they're within the threshold,
        # (3) that the distances are correct, and (4) that the filepath is the same as the query.
        assert len(matches) > 0
        assert matches["distance"].lt(threshold).all()
        assert (matches["metadata.hash"].apply(lambda h: hasher.compute_distance(query_hash, np.frombuffer(h, hasher.dtype))) == matches["distance"]).all()
        assert matches["metadata.filepath"].eq(query_filepath).all()
        con.close()
