#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

cd ${work_dir}
if [[ -e venv ]]; then
  echo "Please remove a previously virtual environment folder '${work_dir}/venv'."
  exit
fi

# Create virtual environment
virtualenv venv -p python3 --prompt="(deep=fr) "
echo "export PYTHONPATH=\$PYTHONPATH:${work_dir}" >> venv/bin/activate
. venv/bin/activate
pip install -r ${work_dir}/requirements.txt


echo
echo "===================================================="
echo "To start to work, you need to activate a virtualenv:"
echo "$ . venv/bin/activate"
echo "===================================================="
