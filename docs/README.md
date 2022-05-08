# Docs

## For Users

View the docs at https://strangequark1041.github.io/samarium/

## For  Devs

Samarium documentation is build using [Doxygen](https://doxygen.nl), which is required to be installed

Build documentation by enabling the cmake flag `"SAMARIUM_BUILD_DOCS"`

```sh
-DSAMARIUM_BUILD_DOCS=ON
```

or to build ony the docs and not the library

```sh
cmake --preset=default
cmake --build build --target docs
```
****
Documentation is built as html in `build/docs`

To view, open [`build/docs/index.html`](build/docs/index.html) in a browser
