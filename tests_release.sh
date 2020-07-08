#!/bin/bash

# Test nightly release: ./tests_release.sh
# Test stable release: ./tests_release.sh --type stable
# Test using PyEnv: ./tests_release.sh --pyenv true

# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x

# Flags
RELEASE_TYPE=nightly
USE_PYENV=false
TEST_COLABS=false

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "--type [nightly|stable]"
  exit 1
fi

while [[ $# -gt -0 ]]; do
  key="$1"
  echo $key
  echo $2
  case $key in
      --type)
      RELEASE_TYPE="$2" # Type of release stable or nightly
      shift
      ;;
    --pyenv)
      USE_PYENV="$2"  # true to use pyenv (Being deprecated)
      shift
      ;;
    --test_colabs)
      TEST_COLABS="$2"  # true to test colabs after build
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      ;;
  esac
  shift # past argument or value
done

PYTHON_VERSION="3.6.1"

run_tests() {
  echo "run_tests:"
  echo "    type:${RELEASE_TYPE}"
  echo "    pyenv:${USE_PYENV}"
  echo "    test_colabs:${TEST_COLABS}"

  if [ "$USE_PYENV" = "true" ]; then
    # Sets up system to use pyenv instead of existing python install.
    if ! stat -t ~/.pyenv/versions/${PYTHON_VERSION}/lib/libpython*m.so > /dev/null 2>&1; then
      # Uninstall current version if there's no libpython file.
      yes | pyenv uninstall $PYTHON_VERSION
    fi
    # We need pyenv to build/install a libpython3.Xm.so file for reverb.
    PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install -s $PYTHON_VERSION
    pyenv global $PYTHON_VERSION
  fi

  TMP=$(mktemp -d)
  # Creates and activates a virtualenv to run the build and unittests in.
  VENV_PATH=${TMP}/virtualenv/$1
  virtualenv -p ~/.pyenv/versions/${PYTHON_VERSION}/bin/python "${VENV_PATH}"
  source ${VENV_PATH}/bin/activate

  # Print the version of python
  python --version
  which pip

  # TensorFlow is not set as a dependency of TF-Agents because there are many
  # different TensorFlow versions a user might want and installed.
  if [ "$RELEASE_TYPE" = "nightly" ]; then
    pip install tf-nightly
    pip install dm-reverb-nightly

    # Run the tests
    python setup.py test

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/
  elif [ "$RELEASE_TYPE" = "stable" ]; then
    pip install tensorflow==2.1.0

    # Run the tests
    python setup.py test --release

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/ --release
  else
    echo "Error unknown --type only [nightly|stable]"
    exit
  fi

  pip install ${WHEEL_PATH}/tf_agents*.whl
  # Simple import test. Move away from repo directory so "import tf_agents"
  # refers to the installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tf_agents')

  # Tests after this run outside the virtual env and depend on packages
  # installed at the system level.
  deactivate

  # Copies wheel out of tmp to root of repo so it can be more easily uploaded
  # to pypi as part of the stable release process.
  cp ${WHEEL_PATH}/tf_agents*.whl ./

  # Testing the Colabs requires packages beyond what is needed to build and
  # unittest TF-Agents, e.g. Jupiter Notebook. It is assumed the base system
  # will have these required packages, which are part of the TF-Agents docker.
  if [ "$TEST_COLABS" = "true" ]; then
    pip install ${WHEEL_PATH}/tf_agents*.whl
    python ./tools/test_colabs.py
  fi

}

if ! which cmake > /dev/null; then
   echo -e "cmake not found! needed for atari_py tests. Install? (y/n) \c"
   read
   if "$REPLY" = "y"; then
      sudo apt-get install -y cmake zlib1g-dev
   fi
fi

# Build and run tests.
run_tests

