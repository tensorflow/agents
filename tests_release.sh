#!/bin/bash

# Test nightly release: ./test_release.sh nightly
# Test stable release: ./test_release.sh stable

# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "test_release [nightly|stable]"
  exit 1
fi

run_tests() {
  echo "run_tests $1 $2"
  TMP=$(mktemp -d)
  # Create and activate a virtualenv to specify python version and test in
  # isolated environment. Note that we don't actually have to cd'ed into a
  # virtualenv directory to use it; we just need to source bin/activate into the
  # current shell.
  VENV_PATH=${TMP}/virtualenv/$1
  virtualenv -p "$1" "${VENV_PATH}"
  source ${VENV_PATH}/bin/activate


  # TensorFlow isn't a regular dependency because there are many different pip
  # packages a user might have installed.
  if [[ $2 == "nightly" ]] ; then
    pip install tf-nightly

    # Run the tests
    python setup.py test

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/
  elif [[ $2 == "stable" ]] ; then
    pip install tensorflow

    # Run the tests
    python setup.py test --release

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/ --release
  else
    echo "Error unknow option only [nightly|stable]"
    exit
  fi

  pip install ${WHEEL_PATH}/tf_agents_*.whl

  # Move away from repo directory so "import tf_agents" refers to the
  # installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tf_agents')

  # Deactivate virtualenv
  deactivate
}

if ! which cmake > /dev/null; then
   echo -e "cmake not found! needed for atari_py tests. Install? (y/n) \c"
   read
   if "$REPLY" = "y"; then
      sudo apt-get install -y cmake zlib1g-dev
   fi
fi

# Test on Python2.7
run_tests "python2.7" $1
# Test on Python3.6
run_tests "python3.6" $1

