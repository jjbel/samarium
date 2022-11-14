Two Particle Collision
======================

Detection
*********

Setup
*****

#. Two spheres of masses :math:`m_1, m_2`
#. Initial velocities :math:`\vec{u_1}, \vec{u_2}`

Handling
********

Equations
---------

By the `Law of Conservation of Linear Momentum <https://en.wikipedia.org/wiki/Momentum#Conservation>`_,

.. math:: m_1u_1 + m_2u_2 = m_1v_1 + m_2v_2
    :label: conserve_momentum

By the definition of the coefficient of restitution,

.. math:: \frac{v_2 - v_1}{u_1 - u_2} = e
    :label: restitution

Solution
--------

From :eq:`conserve_momentum` and :eq:`restitution`

Let,

.. math:: \Delta u = e \cdot (u_1 - u_2)
    :label: delta_u

Then,

.. math:: v_1 = \frac{m_1u_1 + m_2u_2 - \Delta u}{m_1 + m_2}
    :label: v1

.. math:: v_2 = \Delta u + v_1
    :label: v2
