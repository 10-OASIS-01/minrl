"""
Tests Module
Created by: 10-OASIS-01
Date: 2025-02-08 10:40:01 UTC

Contains test cases for all project components.
"""

import unittest


def run_all_tests():
    """Run all test cases in the package"""
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all_tests()