#!/bin/bash

# Install pypi from test: ./test_nightly_pypi.sh test
# Install pypi from pypi: ./test_nightly_pypi.sh official

# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "test_nightly_pypi [test|official]"
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
  if [[ $2 == "test" ]] ; then
    pip install tf-nightly

    # Install pypi package from test.pypi
    pip install --index-url https://test.pypi.org/simple/ tf-agents-nightly

  elif [[ $2 == "official" ]] ; then
    pip install tf-nightly

    # Install pypi package
    pip install tf-agents-nightly
  else
    echo "Error unknow option only [test|official]"
    exit
  fi

  # Move away from repo directory so "import tf_agents" refers to the
  # installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tf_agents')

  # Deactivate virtualenv
  deactivate
}

# Test on Python2.7
run_tests "python2.7" $1
# Test on Python3.6
run_tests "python3.6" $1

