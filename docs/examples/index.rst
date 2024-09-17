Examples
========

These examples are included with the library in ``/examples``

..  toctree::
    softbody
    hilbert_curve

Fourier Series
--------------

Create a complex [Fourier Series](https://en.wikipedia.org/wiki/Fourier_series) to draw a `target_shape` which returns the shape when queried from 0 to 1

Flow Field Noise
----------------

Move particles around by placing them on a grid of forces (a Flow Field)


Mandelbrot
----------

Render a Mandelbrot Set

Highlights: panning, zooming, drawing to pixels

## Gas Maxwell Boltzmann Distribution

The speeds of particles of a gas approach a [Maxwell Boltzmann Distribution](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution)

Highlights: particle simulation, plotting a graph

Poisson Disc Sampling
---------------------

Distribute points evenly in a plane

Highlights: `poisson_disc::uniform`

Prime Spiral
------------

Numbers make spirals when plotted in polar coordinates

Inspired by <https://youtu.be/EK32jo7i5LQ>

Hightlights: `Vector2::from_polar`, panning, zooming

Second Order Dynamics
---------------------

Use the `SecondOrderDynamics` struct to make a point follow the cursor.

Softbody
--------

Full-fledged softbody simulation using damped springs


[](https://user-images.githubusercontent.com/83468982/178472984-8cd83808-bfb2-478b-8a5e-3d45782f2c7d.mp4)
