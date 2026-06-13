"""
Pytest tests for the linear algebra library.
Only one test is implemented; the rest are left as practice exercises.
"""

import math
import pytest

from lab1.lib import (
    Vector,
    Matrix,
    create_identity_matrix,
    create_zero_matrix,
    create_zero_vector,
)


def test_vector_creation():
    vec = Vector([1.0, 2.0, 3.0])

    assert len(vec) == 3
    assert vec[0] == 1.0
    assert vec[1] == 2.0
    assert vec[2] == 3.0


def test_vector_addition():
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])
    result = v1 + v2
    assert result == Vector([5.0, 7.0, 9.0])


def test_vector_subtraction():
    v1 = Vector([4.0, 5.0, 6.0])
    v2 = Vector([1.0, 2.0, 3.0])
    result = v1 - v2
    assert result == Vector([3.0, 3.0, 3.0])


def test_vector_scalar_multiplication():
    v = Vector([1.0, 2.0, 3.0])
    result = v * 3
    assert result == Vector([3.0, 6.0, 9.0])
    result2 = 2 * v
    assert result2 == Vector([2.0, 4.0, 6.0])


def test_vector_magnitude():
    v = Vector([3.0, 4.0])
    assert v.magnitude() == pytest.approx(5.0)

    v2 = Vector([1.0, 1.0, 1.0])
    assert v2.magnitude() == pytest.approx(math.sqrt(3))


def test_vector_dot_product():
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])
    assert v1.dot(v2) == pytest.approx(32.0)


def test_vector_normalization():
    v = Vector([3.0, 4.0])
    n = v.normalize()
    assert n.magnitude() == pytest.approx(1.0)
    assert n == Vector([0.6, 0.8])


def test_vector_distance():
    v1 = Vector([0.0, 0.0])
    v2 = Vector([3.0, 4.0])
    assert v1.distance(v2) == pytest.approx(5.0)


def test_vector_dimension_mismatch():
    v1 = Vector([1.0, 2.0])
    v2 = Vector([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        _ = v1 + v2
    with pytest.raises(ValueError):
        _ = v1 - v2
    with pytest.raises(ValueError):
        v1.dot(v2)


def test_vector_equality():
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([1.0, 2.0, 3.0])
    v3 = Vector([1.0, 2.0, 4.0])
    assert v1 == v2
    assert v1 != v3


def test_matrix_creation():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.rows == 2
    assert m.cols == 2
    assert m[0] == [1.0, 2.0]
    assert m[1] == [3.0, 4.0]


def test_matrix_addition():
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = m1 + m2
    assert result == Matrix([[6.0, 8.0], [10.0, 12.0]])


def test_matrix_subtraction():
    m1 = Matrix([[5.0, 6.0], [7.0, 8.0]])
    m2 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    result = m1 - m2
    assert result == Matrix([[4.0, 4.0], [4.0, 4.0]])


def test_matrix_scalar_multiplication():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    result = m * 2
    assert result == Matrix([[2.0, 4.0], [6.0, 8.0]])
    result2 = 3 * m
    assert result2 == Matrix([[3.0, 6.0], [9.0, 12.0]])


def test_matrix_multiplication():
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = m1 @ m2
    assert result == Matrix([[19.0, 22.0], [43.0, 50.0]])


def test_matrix_vector_multiplication():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    v = Vector([1.0, 1.0])
    result = m @ v
    assert result == Vector([3.0, 7.0])


def test_matrix_transpose():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t = m.transpose()
    assert t == Matrix([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert t.rows == 3
    assert t.cols == 2


def test_matrix_determinant_2x2():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.determinant() == pytest.approx(-2.0)


def test_matrix_determinant_3x3():
    m = Matrix([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
    ])
    assert m.determinant() == pytest.approx(-3.0)


def test_matrix_inverse_2x2():
    m = Matrix([[4.0, 7.0], [2.0, 6.0]])
    inv = m.inverse()
    identity = m @ inv
    expected = create_identity_matrix(2)
    assert identity == expected


def test_matrix_inverse_singular():
    m = Matrix([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(ValueError):
        m.inverse()


def test_matrix_trace():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.trace() == pytest.approx(5.0)

    m2 = Matrix([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    assert m2.trace() == pytest.approx(6.0)


def test_matrix_dimension_mismatch_addition():
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError):
        _ = m1 + m2


def test_matrix_dimension_mismatch_multiplication():
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError):
        _ = m1 @ m2


def test_create_identity_matrix():
    identity = create_identity_matrix(3)
    assert identity.rows == 3
    assert identity.cols == 3
    assert identity[0] == [1.0, 0.0, 0.0]
    assert identity[1] == [0.0, 1.0, 0.0]
    assert identity[2] == [0.0, 0.0, 1.0]


def test_create_zero_matrix():
    m = create_zero_matrix(2, 3)
    assert m.rows == 2
    assert m.cols == 3
    for i in range(2):
        for j in range(3):
            assert m[i][j] == 0.0


def test_create_zero_vector():
    v = create_zero_vector(4)
    assert len(v) == 4
    for i in range(4):
        assert v[i] == 0.0


def test_empty_vector_creation():
    with pytest.raises(ValueError):
        Vector([])


def test_zero_vector_normalization():
    v = Vector([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        v.normalize()


def test_divide_by_zero():
    v = Vector([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        _ = v / 0

    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        _ = m / 0


def test_non_square_matrix_determinant():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError):
        m.determinant()


def test_non_square_matrix_inverse():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError):
        m.inverse()
