Installation
============

.. code-block:: sh
    :linenos:

    pip install conan --upgrade --quiet

    conan profile new default --detect
    conan profile update settings.compiler.libcxx=libstdc++11 default # if on linux
    
    git clone https://github.com/strangeQuark1041/samarium.git
    conan create samarium  -b missing -pr default -pr:b=default

.. asciinema:: 0ZgfSBZKOAZQYE4ngGcoHCReo
