# DIRACOS2

Python 3 DIRACOS installer built using [conda constructor](https://github.com/conda/constructor) and [conda-forge](https://conda-forge.org/).

## Table of contents

* [Installing DIRACOS2](#installing-diracos2)
* [Advanced documentation](#advanced-documentation)
    * [Building the installer](#building-the-installer)
    * [Testing the installer](#testing-the-installer)
    * [Making a release](#making-a-release)
    * [Building customised packages](#building-customised-packages)
    * [Miscellaneous remarks](#miscellaneous-remarks)
* [Troubleshooting](#troubleshooting)
    * [Duplicate files](#duplicate-files)

## Installing DIRACOS2

These instructions will install the latest release of DIRACOS in a folder named `diracos`. To install a specific version replace `/latest/download/` in the URL with a version like `/download/2.0/`.

```bash
curl -LO https://github.com/DIRACGrid/DIRACOS2/releases/latest/download/DIRACOS-Linux-x86_64.sh
bash DIRACOS-Linux-x86_64.sh
```

It can then be activated in a similar way to version 1 of DIRACOS, by calling `source "$PWD/diracos/diracosrc"`.
Further usage instructions are shown after installation.

## Advanced documentation

### Building the installer

The DIRACOS installer is a self-extracting shell script that is generated using [conda constructor](https://github.com/conda/constructor). This can be installed using any `conda` installation. If you don't have a local conda installation, you can use the following steps on Linux:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $PWD/miniforge
# This line is needed every time you wish to use miniforge in a new shell
eval "$($PWD/miniforge/bin/conda shell.bash hook)"
conda install constructor
```

Once you have `constructor` installed, a new installer can be generating from the `construct.yaml` using:

```bash
constructor . --platform=linux-64
```

The packages included are defined in the `construct.yaml` file, see the [upstream documentation](https://github.com/conda/constructor/blob/master/CONSTRUCT.md) for more details.

### Testing the installer

Basic tests of the installer can be ran using an arbitrary docker image by running:

```bash
scripts/run_basic_tests.sh DOCKER_IMAGE DIRACOS_INSTALLER_FILENAME
```

### Making a release

To ensure reproducibility, releases are made from build artifacts from previous pipelines and are tagged using GitHub actions by triggering the [Create release](https://github.com/DIRACGrid/DIRACOS2/actions/workflows/release.yaml?query=workflow%3A%22Create+release%22) workflow. This workflow runs [`scripts/make_release.py`](https://github.com/DIRACGrid/DIRACOS2/blob/main/scripts/make_release.py) and has the following optional parameters:

* **Run ID**: The GitHub Actions workflow run ID. If not given, defaults to the most recent build of the `main` branch.
* **Version number**: A [PEP-440](https://www.python.org/dev/peps/pep-0440/) compliant version number. If not given, defaults to the contents the contents of version field in the `construct.yaml` rounded to the next full release (i.e. `2.4a5` becomes `2.4` and `2.1` remains unchanged). If a pre-release is explicitly give, it will be marked as a pre-release in GitHub and won't affect the `latest` alias.

If the release process fails, a draft release might be left in GitHub. After the issue has been fixed this can be safely deleted before rerunning the CI.

After the release is made a commit will be pushed to master to bump the DIRACOS release number in `construct.yaml` to the next alpha release. If the next alpha release is older than the current contents of `construct.yaml`, this step is skipped.

### Building customised packages

See [`management/conda-recipes`](https://github.com/DIRACGrid/management/tree/master/conda-recipes).

### Miscellaneous remarks

DIRACOS2 is a smaller wrapper around [conda constructor](https://github.com/conda/constructor) which generates a self-extracting installer that is suitable for DIRAC's needs based on packages which are distributed by [conda-forge](https://conda-forge.org/). As such, most components are documented by their respective projects. The following is a list of remarks that are specific DIRACOS2:

* The build-and-test workflow is ran on every push event as well as once per day. GitHub Actions limitations will cause the nightly build to be disabled if there is no activity in the repository for 60 days. If this happens, it can be easily re-enabled by pushing an empty commit to master.
* The DIRACOS installer follows the common pattern of being a shell script that exits followed by an arbitrary binary blob. In principle it is possible to create other kinds of installer with `constructor` however these are no deemed to be of interest at the current time.
* Similarly to the original DIRACOS, releases are made by assigning a version number to a previous CI build. As `constructor` embeds the version number into the installer at build time, this means the version of the installed has to be edited for the binary artifact. Fortunately this is only embedded in the shell script at the start of the file, so it's a relatively simple string manipulation to modify it. This is done using in the `main()` function of [`scripts/make_release.py`](https://github.com/DIRACGrid/DIRACOS2/blob/main/scripts/make_release.py). As this is not an officially supported feature of constructor it is prone to changing at anytime.

## Troubleshooting

This section contains a list of know issues that can occur.

### Duplicate files

This error is of the form:

```
File 'include/event.h' found in multiple packages: libevent-2.1.10-hcdb4288_2.tar.bz2, libev-4.33-h516909a_0.tar.bz2
```

When this happens, the offending packages should be fixed upstream in conda-forge. As a temporary workaround, the `ignore_duplicate_files` key in `construct.yaml` can be changed to `true`.
