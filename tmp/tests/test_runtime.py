import unittest

from vla.runtime import Runtime, adaptation_cells, training_steps


class RuntimeTest(unittest.TestCase):
    def test_screening_plan_is_one_small_cell(self):
        runtime = Runtime("mps", 2, 4, "no", 5, (0,), (5,), (0,))
        self.assertEqual(list(adaptation_cells(runtime)), [(0, 0, 5)])
        self.assertEqual(training_steps(runtime, 5), 1500)

    def test_cuda_plan_is_the_required_matrix(self):
        runtime = Runtime("cuda", 32, 8, "bf16", 20, (0, 1, 2), (5, 10, 25), (0, 1))
        self.assertEqual(len(list(adaptation_cells(runtime))), 18)
        self.assertEqual(training_steps(runtime, 25), 7500)


if __name__ == "__main__":
    unittest.main()
