import random
import unittest

import numpy as np

from pipeline.lane_context import adjust, bin_of, evaluate, fit_compatibility, lateral, neighbors_of


def car(camera, file, x, truth, probs, width=.06):
    return {'camera': camera, 'file': file, 'box': [x, .5, x + width, .56], 'probs': probs, 'truth': truth}


class LaneContextTests(unittest.TestCase):
    def test_lane_offset_is_in_car_widths(self):
        self.assertAlmostEqual(lateral([.1, .5, .2, .6], [.3, .5, .4, .6]), 2)
        self.assertAlmostEqual(lateral([.45, .2, .47, .22], [.49, .2, .51, .22]), 2)
        self.assertEqual(bin_of([.1, .5, .2, .6], [.15, .5, .25, .6]), 0)
        self.assertEqual(bin_of([.1, .5, .15, .6], [.6, .5, .65, .6]), 4)

    def test_compatibility_is_a_conditional_distribution_per_offset(self):
        table, marginal = fit_compatibility([(0, 1, 1)] * 20 + [(3, 0, 1)] * 20)
        self.assertTrue(np.allclose(table.sum(1), 1))
        self.assertAlmostEqual(marginal.sum(), 1)
        self.assertGreater(table[0, 1, 1], .7, 'same lane: rear follows rear')
        self.assertGreater(table[3, 0, 1], .4, 'across the road: front opposite rear')

    def test_neighbors_shift_an_unsure_car_and_alpha_zero_leaves_it(self):
        table, marginal = fit_compatibility([(0, 1, 1)] * 20 + [(0, 0, 0)] * 20)
        unsure = [.5, .42, .08]
        self.assertEqual(np.argmax(adjust(unsure, [(0, [0, 1, 0])], table, marginal, 1)), 1)
        self.assertTrue(np.allclose(adjust(unsure, [(0, [0, 1, 0])], table, marginal, 0), np.array(unsure) / sum(unsure)))
        self.assertTrue(np.allclose(adjust(unsure, [], table, marginal, 1), np.array(unsure) / sum(unsure)))

    def test_a_parked_car_in_another_frame_is_not_its_own_neighbor(self):
        target = car('1', 'a.jpg', .4, 0, [.5, .4, .1])
        same_frame = car('1', 'a.jpg', .5, None, [0, 1, 0])
        parked = {**car('1', 'b.jpg', .401, None, [0, 1, 0]), 'history': True}
        moved = {**car('1', 'c.jpg', .43, None, [0, 1, 0]), 'history': True}
        cars = [target, same_frame, parked, moved]
        self.assertEqual(len(neighbors_of(target, cars)), 2)
        self.assertEqual(len(neighbors_of(target, cars, history=False)), 1)

    def test_camera_grouped_evaluation_gains_where_lanes_decide_facing(self):
        # Each camera: a rear-facing lane on the left, a front-facing lane on the right.
        # The crop model is right 70% of the time; its neighbors in the lane are informative.
        rng = random.Random(3)
        by_camera = {}
        for c in range(60):
            cars = []
            for i in range(8):
                truth = 1 if i % 2 else 0
                x = (.2 if truth else .6) + rng.uniform(-.02, .02)
                right = rng.random() < .7
                probs = [.2, .2, .1]; probs[truth if right else 1 - truth] = .5
                cars.append(car(str(c), f'{c}.jpg', x, truth if i < 4 else None, probs))
            by_camera[str(c)] = cars
        report = evaluate(by_camera, folds=5)
        self.assertGreater(report['accuracy']['lane_context'], report['accuracy']['crop_model'] + .05)
        self.assertGreater(min(report['alphas']), 0)
        self.assertEqual(report['labels'], 240)
