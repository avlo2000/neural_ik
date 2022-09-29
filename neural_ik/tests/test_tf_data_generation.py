from unittest import TestCase
import tempfile

from data.data_io import write_csv, read_csv
from data.tf_kin_data import generate_with_theta_seed, rawdata_to_dataset
from tf_kinematics import kinematic_models


class Test(TestCase):
    def test_generate_with_theta_seed(self):
        kin = kinematic_models.kuka_robot(1)
        size = 100
        seed_multiplier = 0.1

        feature_names, raw_data = generate_with_theta_seed(kin, size, seed_multiplier)

        tmp = tempfile.TemporaryFile('w+t')
        write_csv(feature_names, raw_data, tmp)
        tmp.seek(0)
        feature_names_read, raw_data_read = read_csv(tmp)
        tmp.close()
        self.assertEqual(feature_names, feature_names_read)
        for raw in raw_data_read:
            self.assertIn(raw, raw_data)
        self.assertEqual(len(raw_data), len(raw_data_read))

    def test_rawdata_to_dataset(self):
        kin = kinematic_models.kuka_robot(1)
        size = 100
        seed_multiplier = 0.1

        feature_names, raw_data = generate_with_theta_seed(kin, size, seed_multiplier)
        thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
        self.assertEqual(len(thetas), len(thetas_seed))
        self.assertEqual(len(thetas), len(iso_transforms))
