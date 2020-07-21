#!/bin/bash

# Test nightly release: ./tests_release.sh
# Test stable release: ./tests_release.sh --type stable
# Test using PyEnv: ./tests_release.sh --pyenv true

# Exits if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x

# Flags
RELEASE_TYPE=nightly
USE_PYENV=false
TEST_COLABS=false
TF_INSTALL=false
TF_DEP_OVERRIDE=false
REVERB_INSTALL=false
REVERB_DEP_OVERRIDE=false
TFP_INSTALL=false
TFP_DEP_OVERRIDE=false

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "--type [nightly|stable]"
  echo "--tf_dep_override     [Required tensorflow version to pass to setup.py."
  echo "                       Examples: tensorflow==2.3.0rc0  or tensorflow>=2.3.0]"
  echo "--reverb_dep_override [Required reverb version to pass to setup.py."
  echo "                        Examples: dm-reverb==0.1.0rc0  or dm-reverb>=0.1.0]"
  echo "--tfp_dep_override    [Required tensorflow version to pass to setup.py."
  echo "                       Examples: tensorflow-probability==0.11.0rc0]"
  echo "--tf_install          [Version of TensorFlow to install]"
  echo "--reverb_install      [Version of Reverb to install]"
  echo "--tfp_install         [Version of TensorFlow probability to install]"
  echo "--test_colabs         [true to run colab tests.]"
  echo "--pyenv               [true, use pyenv (Being deprecated)]"
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
      USE_PYENV="$2"  # If true, use pyenv (Being deprecated)
      shift
      ;;
    --test_colabs)
      TEST_COLABS="$2"  # If true, test colabs after build
      shift
      ;;
    --tf_install)
      TF_INSTALL="$2"  # Install this version of TensorFlow.
      shift
      ;;
    --tf_dep_override)
      TF_DEP_OVERRIDE="$2"  # Setup.py is told this is the required tensorflow.
      shift
      ;;
    --reverb_install)
      REVERB_INSTALL="$2"  # Install this version of Reverb.
      shift
      ;;
    --reverb_dep_override)
      REVERB_DEP_OVERRIDE="$2"  # Setup.py is told this is the required reverb.
      shift
      ;;
    --tfp_install)
      TFP_INSTALL="$2"  # Install this version of tf-probability.
      shift
      ;;
    --tfp_dep_override)
      TFP_DEP_OVERRIDE="$2"  # Setup.py is told this is the required tf-probability.
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      ;;
  esac
  shift # past argument or value
done

install_optional_dependencies() {
  if [ "$TF_INSTALL" != "false" ]; then
    pip install $TF_INSTALL
  else
    pip install $1
  fi

  if [ "$REVERB_INSTALL" != "false" ]; then
    pip install $REVERB_INSTALL
  else
    pip install $2
  fi

  if [ "$TFP_INSTALL" != "false" ]; then
    pip install $TFP_INSTALL
  else
    pip install $3
  fi
}

run_tests() {
  echo "run_tests:"
  echo "    type:${RELEASE_TYPE}"
  echo "    pyenv:${USE_PYENV}"
  echo "    test_colabs:${TEST_COLABS}"
  echo "    tf_installs:${TF_INSTALL}"
  echo "    reverb_install:${REVERB_INSTALL}"
  echo "    tfp_install:${TFP_INSTALL}"
  echo "    tf_dep_override:${REVERB_DEP_OVERRIDE}"
  echo "    reverb_dep_override:${REVERB_DEP_OVERRIDE}"
  echo "    tfp_dep_override:${TFP_DEP_OVERRIDE}"

  PYTHON_BIN_PATH=$(which python)
  if [ "$USE_PYENV" = "true" ]; then
    PYTHON_VERSION="3.6.1"
    # Sets up system to use pyenv instead of existing python install.
    if ! stat -t ~/.pyenv/versions/${PYTHON_VERSION}/lib/libpython*m.so > /dev/null 2>&1; then
      # Uninstall current version if there's no libpython file.
      yes | pyenv uninstall $PYTHON_VERSION
    fi
    # We need pyenv to build/install a libpython3.Xm.so file for reverb.
    PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install -s $PYTHON_VERSION
    pyenv global $PYTHON_VERSION
    PYTHON_BIN_PATH=~/.pyenv/versions/${PYTHON_VERSION}/bin/python
  fi

  TMP=$(mktemp -d)
  # Creates and activates a virtualenv to run the build and unittests in.
  VENV_PATH=${TMP}/virtualenv/$1
  virtualenv -p "${PYTHON_BIN_PATH}" "${VENV_PATH}"
  source ${VENV_PATH}/bin/activate

  # Print the version of python
  python --version
  which pip

  # Extra args to pass to setup.py
  EXTRA_ARGS=""
  if [ "$TF_DEP_OVERRIDE" != "false" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --tf-version ${TF_DEP_OVERRIDE}"
  fi
  if [ "$REVERB_DEP_OVERRIDE" != "false" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --reverb-version ${REVERB_DEP_OVERRIDE}"
  fi
  if [ "$TFP_DEP_OVERRIDE" != "false" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --tfp-version ${TFP_DEP_OVERRIDE}"
  fi
  # TensorFlow is not set as a dependency of TF-Agents because there are many
  # different TensorFlow versions a user might want and installed.
  if [ "$RELEASE_TYPE" = "nightly" ]; then
    install_optional_dependencies "tf-nightly" "dm-reverb-nightly" "tfp-nightly"

    # Run the tests
    python setup.py test $EXTRA_ARGS

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/ $EXTRA_ARGS
  elif [ "$RELEASE_TYPE" = "stable" ]; then
    install_optional_dependencies "tensorflow" "dm-reverb" "tensorflow-probability"
    # Run the tests
    python setup.py test --release $EXTRA_ARGS

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/ --release $EXTRA_ARGS
  else
    echo "Error unknown --type only [nightly|stable]"
    exit
  fi

  WHL_PATH=$(find ${WHEEL_PATH} -path \*tf_agents\*.whl)
  pip install ${WHL_PATH}
  # Simple import test. Move away from repo directory so "import tf_agents"
  # refers to the installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tf_agents')

  # Tests after this run outside the virtual env and depend on packages
  # installed at the system level.
  deactivate

  # Copies wheel out of tmp to root of repo so it can be more easily uploaded
  # to pypi as part of the stable release process.
  cp ${WHEEL_PATH}tf_agents*.whl ./

  # Testing the Colabs requires packages beyond what is needed to build and
  # unittest TF-Agents, e.g. Jupiter Notebook. It is assumed the base system
  # will have these required packages, which are part of the TF-Agents docker.
  if [ "$TEST_COLABS" = "true" ]; then
    pip install ${WHL_PATH}[reverb]
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
