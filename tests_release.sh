#!/bin/bash

# Test nightly release: ./tests_release.sh
# Test stable release: ./tests_release.sh --type stable

# Exits if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x

# Flags
RELEASE_TYPE=nightly
TEST_COLABS=false
TF_INSTALL=false
TF_DEP_OVERRIDE=false
REVERB_INSTALL=false
REVERB_DEP_OVERRIDE=false
TFP_INSTALL=false
TFP_DEP_OVERRIDE=false
RLDS_INSTALL=false
RLDS_DEP_OVERRIDE=false
PYTHON_VERSION=python3
BROKEN_TESTS=false

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "--type [nightly|stable]"
  echo "--tf_dep_override     [Required tensorflow version to pass to setup.py."
  echo "                       Examples: tensorflow==2.3.0rc0  or tensorflow>=2.3.0]"
  echo "--reverb_dep_override [Required reverb version to pass to setup.py."
  echo "                        Examples: dm-reverb==0.1.0rc0  or dm-reverb>=0.1.0]"
  echo "--tfp_dep_override    [Required tensorflow version to pass to setup.py."
  echo "                       Examples: tensorflow-probability==0.11.0rc0]"
  echo "--rlds_dep_override   [Version of RLDS to install.]"
  echo "                       Examples: rlds==0.1.3]"
  echo "--tf_install          [Version of TensorFlow to install]"
  echo "--reverb_install      [Version of Reverb to install]"
  echo "--tfp_install         [Version of TensorFlow probability to install]"
  echo "--rlds_install         [Version of TensorFlow probability to install]"
  echo "--test_colabs         [true to run colab tests.]"
  echo "--python_version    [python3.7(default), Python binary to use.]"
  echo "--broken_tests   [Broken tests file]"
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
    --rlds_install)
      RLDS_INSTALL="$2"  # Install this version of rlds.
      shift
      ;;
    --rlds_dep_override)
      RLDS_DEP_OVERRIDE="$2"  # Setup.py is told this is the required rlds.
      shift
      ;;
    --python_version)
      PYTHON_VERSION="$2"  # Python binary to use for the build.
      shift
      ;;
    --broken_tests)
      BROKEN_TESTS="$2"  # Python binary to use for the build.
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
    $PYTHON_VERSION -mpip install $TF_INSTALL
  else
    $PYTHON_VERSION -mpip install $1
  fi

  if [ "$REVERB_INSTALL" != "false" ]; then
    $PYTHON_VERSION -mpip install $REVERB_INSTALL
  else
    $PYTHON_VERSION -mpip install $2
  fi

  if [ "$TFP_INSTALL" != "false" ]; then
    $PYTHON_VERSION -mpip install $TFP_INSTALL
  else
    $PYTHON_VERSION -mpip install $3
  fi

  if [ "$RLDS_INSTALL" != "false" ]; then
    $PYTHON_VERSION -mpip install $RLDS_INSTALL
  else
    $PYTHON_VERSION -mpip install $4
  fi
}

run_tests() {
  echo "run_tests:"
  echo "    type:${RELEASE_TYPE}"
  echo "    python_version:${PYTHON_VERSION}"
  echo "    test_colabs:${TEST_COLABS}"
  echo "    tf_installs:${TF_INSTALL}"
  echo "    reverb_install:${REVERB_INSTALL}"
  echo "    tfp_install:${TFP_INSTALL}"
  echo "    tf_dep_override:${REVERB_DEP_OVERRIDE}"
  echo "    reverb_dep_override:${REVERB_DEP_OVERRIDE}"
  echo "    tfp_dep_override:${TFP_DEP_OVERRIDE}"
  echo "    broken_tests:${BROKEN_TESTS}"

  PYTHON_BIN_PATH=$(which $PYTHON_VERSION)
  TMP=$(mktemp -d)
  # Creates and activates a virtualenv to run the build and unittests in.
  VENV_PATH=${TMP}/virtualenv/$1
  virtualenv -p "${PYTHON_BIN_PATH}" "${VENV_PATH}"
  source ${VENV_PATH}/bin/activate

  # Print the version of python
  $PYTHON_VERSION --version
  # Used by pip_pig.sh
  export PYTHON_VERSION
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
  if [ "$RLDS_DEP_OVERRIDE" != "false" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --rlds-version ${RLDS_DEP_OVERRIDE}"
  fi
  if [ "$BROKEN_TESTS" != "false" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --broken_tests ${BROKEN_TESTS}"
  fi

  # TensorFlow is not set as a dependency of TF-Agents because there are many
  # different TensorFlow versions a user might want and installed.
  if [ "$RELEASE_TYPE" = "nightly" ]; then
    # TODO(b/224850217): rlds does not have nightly builds yet.
    install_optional_dependencies "tf-nightly" "dm-reverb-nightly" "tfp-nightly" "rlds"

    # Run the tests
    $PYTHON_VERSION setup.py test $EXTRA_ARGS

    # Builds tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/ $EXTRA_ARGS
  elif [ "$RELEASE_TYPE" = "stable" ]; then
    install_optional_dependencies "tensorflow" "dm-reverb" "tensorflow-probability" "rlds"
    # Run the tests
    $PYTHON_VERSION setup.py test --release $EXTRA_ARGS

    # Builds tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/ --release $EXTRA_ARGS
  else
    echo "Error unknown --type only [nightly|stable]"
    exit
  fi

  WHL_PATH=$(find ${WHEEL_PATH} -path \*tf_agents\*.whl)
  $PYTHON_VERSION -mpip install ${WHL_PATH}
  # Simple import test. Move away from repo directory so "import tf_agents"
  # refers to the installed wheel and not to the local fs.
  (cd $(mktemp -d) && $PYTHON_VERSION -c 'import tf_agents')

  # Tests after this run outside the virtual env and depend on packages
  # installed at the system level.
  deactivate

  # Copies wheel out of tmp to root of repo so it can be uploaded to pypi.
  mkdir -p ./dist
  cp ${WHL_PATH} ./dist/

  # Testing the Colabs is done in a virtualenv due to the docker container used
  # for builds ending up in an unreliable state when installing non-nightly
  # versions for Tensorflow.
  if [ "$TEST_COLABS" = "true" ]; then
    COLAB_TMP=$(mktemp -d)
    COLAB_VENV_PATH=${COLAB_TMP}/virtualenv/$1
    virtualenv -p "${PYTHON_BIN_PATH}" "${COLAB_VENV_PATH}"
    source ${COLAB_VENV_PATH}/bin/activate
    $PYTHON_VERSION -m pip install ${WHL_PATH}[reverb]
    $PYTHON_VERSION -m pip install jupyter ipykernel matplot
    $PYTHON_VERSION ./tools/test_colabs.py
    deactivate
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
