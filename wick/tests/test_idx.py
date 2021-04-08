import unittest

from wick.index import Idx, is_occupied

class IdxTest(unittest.TestCase):
    def test_idx(self):
        i = Idx(0,"occ")
        j = Idx(1,"occ")
        a = Idx(0,"vir")
        b = Idx(1,"vir")
        c = Idx(0,"vir")

        self.assertTrue(i != j)
        self.assertTrue(a != b)
        self.assertTrue(c == a)

    def test_idx_str(self):
        i = Idx(0,"occ")
        b = Idx(1,"vir")

        self.assertTrue(str(i) == "0(occ)")
        self.assertTrue(str(b) == "1(vir)")

    def test_idx_occ(self):
        i = Idx(0,"occ")
        b = Idx(1,"spx")

        self.assertTrue(is_occupied(i))
        self.assertTrue(is_occupied(b, occ=["spx"]))


if __name__ == '__main__':
    unittest.main()
