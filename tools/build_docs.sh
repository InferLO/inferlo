# Genrates local copy of InferLO documentation.
# Inspired by https://github.com/quantumlib/Cirq/blob/master/dev_tools/docs/build-rtd-docs.sh

set -e

# Get the working directory to the repo root.
cd "$(git rev-parse --show-toplevel)"/docs

docs_conf_dir="."
out_dir="${docs_conf_dir}/_build"

# Cleanup pre-existing temporary generated files.
rm -rf "${docs_conf_dir}/generated"

# Cleanup previous output.
rm -rf "${out_dir}"

# Regenerate docs.
sphinx-build -M html "${docs_conf_dir}" "${out_dir}" -W --keep-going

# Cleanup newly generated temporary files.
rm -rf "${docs_conf_dir}/generated"

touch ${docs_conf_dir}/_build/html/.nojekyll

echo
echo Index Page:
echo "file://$(pwd)/${out_dir}/html/index.html"
echo