Hilbert Curve
=============

* Interpolate between different order Hilbert Curves
* Highlights: ``interp``, ``lerp``, ``ease``, seamless looping
* Inspiration: `3Blue1Brown <https://youtu.be/3s7h2MHQtxc>`_, `Coding Train <https://youtu.be/dSK-MW-zuAc>`_

.. video:: https://user-images.githubusercontent.com/83468982/178472984-8cd83808-bfb2-478b-8a5e-3d45782f2c7d.mp4
    :autoplay:
    :loop:

A `Hilbert Curve <https://en.wikipedia.org/wiki/Hilbert_curve>`_ is a `space-filling curve <https://en.wikipedia.org/wiki/Space-filling_curve>`_ : an zero-width line filling 2d space.

It maps the unit square onto the unit interval. That is, every point in the square can be assigned a unique coordinate from 0 to 1.

The true curve has infinite length. In this example we draw successive approximations to the curve, which approach it in the limit as n goes to infinity.
