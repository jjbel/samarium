Installation
============

1. **Install Python**: https://python.org/downloads/

2. **Install Git**: https://git-scm.com/downloads

3. **Install and Setup Conan**: 
    .. code-block:: sh

        python -m pip install conan --upgrade --quiet

        conan profile new default --detect
        # if using gcc:
        conan profile update settings.compiler.libcxx=libstdc++11 default

4. **Install samarium**
    .. code-block:: sh

        git clone https://github.com/strangeQuark1041/samarium.git
        conan create samarium -b missing -pr default -pr:b=default

.. asciinema:: 0ZgfSBZKOAZQYE4ngGcoHCReo
