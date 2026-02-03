### 1. Mapping from Z^2 to N, ie shd include -ve nos

a. If don't want to define simulation rectangle

[Wiki: bijection from Z to N](https://en.wikipedia.org/wiki/Bijection#/media/File:A_bijection_from_the_natural_numbers_to_the_integers.png)

```
for n >= 0:
2n-1 <-> n
2n <-> -n
```

then use [Wiki: Cantor pairing function](https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function)

TODO could also use hilbert curve? then it preserves locality: good for collision checking

b. just use good old row-major indexing
TODO could also do the origin shifting to the float coords first
simple if origin is in middle

x: {-w ... w-1}
y: {-h ... h-1}

TODO without division?
i = (x + w) + (y + h) * w = x + (y + h + 1) * w
x = i % w
y = (i - x) / w - h - 1
