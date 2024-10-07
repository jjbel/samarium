Fourier Shape
=============

* Animate a fourier series of different shapes
* Highlights: ``integrate``, ``sample``, ``Trail``
* Inspiration: `3Blue1Brown <https://youtu.be/r6sGWTCMz2k>`_

.. video:: https://github.com/user-attachments/assets/d870c975-44d4-4624-b122-48129506bbf6
    :autoplay:
    :loop:

A Fourier Series of n terms is an approximation of a given function using sine waves of frequencies upto n.
We can represent shapes (curves) using complex functions. We find the amplitudes of the frequencies forming the function by finding its `Fourier transform <https://en.wikipedia.org/wiki/Fourier_transform>`_

Given a (complex) function :math:`u(t) : [a, b] \to C`
The amplitude of a frequency f as a function of f is given by the fourier transform :math:`\widehat{u}(f)` :

.. math:: \widehat{u}(f) = \int_{a}^{b} u(t) e^{- 2 \pi i f t} dt

We compute this by using the Reimann sum of the integral instead: the function ```math::integrate```
 
To find the function for an arbitrary shape

#. trace over the shape in Blender
#. export as a ``.obj`` file
#. use ``obj_to_pts`` to import and parse the ``obj``
#. use ``ShapeFnFromPts`` to make a function which interpolates between the points and returns a complex number.
