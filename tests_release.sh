#!/bin/bash

# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x

run_tests() {
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
  pip install tensorflow

  # Run the tests
  python setup.py test --release

  # Install tf_agents package.
  WHEEL_PATH=${TMP}/wheel/$1
  ./pip_pkg.sh ${WHEEL_PATH}/ --release

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
run_tests "python2.7"
# Test on Python3.5
run_tests "python3.5"
