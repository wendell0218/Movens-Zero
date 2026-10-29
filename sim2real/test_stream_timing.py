import unittest
from motion_bridge import next_reference_deadline


class TimingTests(unittest.TestCase):
    def test_jitter_does_not_shift_timeline(self):
        deadline = None
        for i in range(15):
            actual = 10 + i * 0.4 + (0.08 if i == 3 else 0)
            deadline = next_reference_deadline(deadline, actual, 20)
            self.assertAlmostEqual(deadline, 10 + (i + 1) * 0.4)

    def test_first_send_anchors_actual_time(self):
        self.assertAlmostEqual(next_reference_deadline(None, 123.456, 20), 123.856)

    def test_final_partial_advance(self):
        self.assertAlmostEqual(next_reference_deadline(20, 20.08, 6), 20.12)

    def test_reject_nonpositive_advance(self):
        with self.assertRaises(ValueError):
            next_reference_deadline(1, 1, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
