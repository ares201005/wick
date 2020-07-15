import unittest
import test_aterm
import test_operators
import test_pair_list
import test_term
import test_term_map
import test_test

def run_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_test.TestTest("test_framework"))

    suite.addTest(test_operators.OperatorTest("test_operator"))
    suite.addTest(test_operators.OperatorTest("test_tensor"))
    suite.addTest(test_operators.OperatorTest("test_sigma"))
    suite.addTest(test_operators.OperatorTest("test_delta"))

    #suite.addTest(test_pair_list.PairListTest("test_p2"))
    #suite.addTest(test_pair_list.PairListTest("test_p4"))
    #suite.addTest(test_pair_list.PairListTest("test_p6"))

    suite.addTest(test_term_map.TermMapTest("test_null"))
    suite.addTest(test_term_map.TermMapTest("test_label"))

    suite.addTest(test_term.TermTest("test_scalar_mul"))
    suite.addTest(test_term.TermTest("test_mul"))
    suite.addTest(test_term.TermTest("test_mul2"))
    suite.addTest(test_term.TermTest("test_term_map"))

    suite.addTest(test_aterm.ATermTest("test_connected"))
    suite.addTest(test_aterm.ATermTest("test_reducible"))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(run_suite())
