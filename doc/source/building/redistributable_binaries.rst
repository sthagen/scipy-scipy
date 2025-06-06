Building redistributable binaries
=================================

*The intended audience for this section is anyone who wants to build SciPy and
deploy it anywhere else than their own machine - from distro packagers to users
who want to build wheels to deploy to their production environment*

When ``python -m build`` or ``pip wheel`` is used to build a SciPy wheel,
that wheel will rely on external shared libraries (at least for BLAS/LAPACK and
a Fortran compiler runtime library, perhaps other libraries). Such wheels
therefore will only run on the system on which they are built. See
`the pypackaging-native content under "Building and installing or uploading
artifacts" <https://pypackaging-native.github.io/meta-topics/build_steps_conceptual/#building-and-installing-or-uploading-artifacts>`__ for more context on that.

A wheel like that is therefore an intermediate stage to producing a binary that
can be distributed. That final binary may be a wheel - in that case, run
``auditwheel`` (Linux), ``delocate`` (macOS), ``delvewheel`` (Windows) or
``repairwheel`` (platform-independent) to vendor the required shared libraries
into the wheel.

The final binary may also be in another packaging format (e.g., a ``.rpm``,
``.deb`` or ``.conda`` package). In that case, there are packaging
ecosystem-specific tools to first install the wheel into a staging area, then
making the extension modules in that install location relocatable (e.g., by
rewriting RPATHs), and then repackaging it into the final package format.


Build and runtime dependencies
------------------------------

The Python build and runtime dependencies that are needed to build SciPy can
be found in the ``pyproject.toml`` metadata. Note that for released versions of
SciPy, dependencies will likely have upper bounds. Each upper bound has
comments above it; packagers are free to remove or loosen those upper bound in
most cases (except for ``numpy``). E.g.::

    # The upper bound on pybind11 is preemptive only
    "pybind11>=2.12.0,<2.13.0",

    #   ...
    #   3. The <2.3 upper bound is for matching the numpy deprecation policy,
    #      it should not be loosened.
    "numpy>=2.0.0rc1,<2.3",

Non-Python build requirements are:

- C, C++ and Fortran compilers
- BLAS and LAPACK libraries
- ``ninja``
- ``pkg-config``

Minimum versions of common compilers are enforced in the top-level
``meson.build`` file. The minimum LAPACK version is currently 3.7.1.
More detailed information on these build dependencies can be found in
:ref:`toolchain-roadmap`.


Using system dependencies instead of vendored sources
-----------------------------------------------------

SciPy contains a *lot* of code that has been vendored from libraries written in
C, C++ and Fortran. This is done for a mix of historical and robustness
reasons. Distro packagers sometimes have reasons to want to *unvendor* this
code, ensuring that they only have a single version of a particular library in
use. We offer a build option, ``-Duse-system-libraries``, to allow them to do
so in a much easier fashion than can be done by manually patching SciPy.

.. code::

    $ python -m build -wnx -Duse-system-libraries=boost.math

This build option causes a dependency lookup *for the exact same version* as is
bundled in SciPy only. A different system version will not be accepted, the
build will then fall back to the vendored sources or error out. This is because
different versions can easily cause failures at build time or at runtime, and
SciPy's CI does not test with any other versions. This may change in the
future, however as of now, using a different version requires patching out the
relevant version check and is strongly discouraged.

Advice for packagers:

1. You should think about this build option as "if I'm already unbundling a
   library for some compelling reason, this option makes that quite a bit
   easier".
2. You should not default to unbundling everything, this will cause diamond
   dependencies and more build complexity for little gain. Most of these
   libraries are not maintained as well as SciPy and/or known to not offer a
   stable API; effectively you'd be adding an extra ``==`` dependency for each
   library you unbundle. Only do that for a good reason. E.g., it allows you to
   drop patches, you consider both SciPy and the vendored library in question
   security-critical, or it's a large gain in binary size.

The build option takes the following values:

- ``none``: Use the source code shipped as part of SciPy, do not look for
  external dependencies (default).
- ``all``: Use external libraries for all components controlled by
  ``use-system-libraries``, and error out if they are not found. Detecting is
  done by Meson, typically first through the ``pkg-config`` and then the
  ``cmake`` method of the ``dependency()`` function.
- ``auto``: Try detecting external libraries, and if they are not found then
  fall back onto vendored sources.
- A comma-separated list of dependency names (e.g., ``boost.math,qhull``). If
  given, uses the ``all`` behavior for the named dependencies (and ``none`` for
  the rest). Supported options are:

  - ``boost.math`` (since 1.16.0)
  - ``qhull`` (since 1.16.0)

It's also valid to combine the generic and named-library options above, for
example ``boost.math,auto`` will apply the ``auto`` behavior to all
dependencies other than ``boost.math``.


Stripping the test suite from a wheel or installed package
----------------------------------------------------------

By default, an installed version of ``scipy`` includes the full test suite.
That test suite, including data files and compiled extension modules that are
test-only, takes up about 4.5 MB in a wheel (for x86-64, as of v1.14.0), and
more than that on disk. In cases where binary size matters, packagers may want
to remove the test suite. As of SciPy 1.14.0, there is a convenient way of
doing this, making use of
`Meson's install tags <https://mesonbuild.com/Installing.html#installation-tags>`__
functionality. It is a one-liner::

    $ python -m build -wnx -Cinstall-args=--tags=runtime,python-runtime,devel

.. note::

   Note that in the above command ``-wnx`` means ``--wheel --no-isolation
   --skip-dependency-check``. It assumes that the packager has already set up
   the build environment, which is usually the case for distro packaging. The
   install tags feature works equally well with isolated builds (e.g. ``pip
   install scipy --no-binary -Cinstall-args=--tags=runtime,python-runtime,devel``).

If you want to produce a separate package for the tests themselves, say under
the name ``scipy-tests``, then edit ``pyproject.toml`` to change the project
name:

.. code:: toml

    [project]
    name = "scipy-tests"

And then build with::

    $ python -m build -wnx -Cinstall-args=--tags=tests

The above would build the whole package twice; in order to rebuild in a cached
fashion, use the ``-Cbuild-dir=build`` build option::

    $     $ # apply patch to change the project name in pyproject.toml
    $ python -m build -wnx -Cbuild-dir=build -Cinstall-args=--tags=tests

The end result will look something like::

    $ ls -lh dist/*.whl
    ...  20M  ...  dist/scipy-1.14.0-cp311-cp311-linux_x86_64.whl
    ...  4,5M ...  dist/scipy_tests-1.14.0-cp311-cp311-linux_x86_64.whl
