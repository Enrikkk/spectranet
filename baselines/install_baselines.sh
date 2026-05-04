#!/usr/bin/env bash
#
# Clone the upstream operator-library repositories used by the SpectraNet
# benchmark.  We do not vendor third-party code; instead we record the exact
# commit pinning each baseline so reviewers can reproduce the same numbers.
#
# Run from the release/ root:
#     bash baselines/install_baselines.sh
#
# Each repository ends up under ./third_party/<name>/.  The first-party
# adapter scripts in baselines/ns_*.py reference these paths via the
# --baseline_root CLI flag.
#
set -euo pipefail

THIRD=${THIRD_PARTY:-./third_party}
mkdir -p "$THIRD"
cd "$THIRD"

clone_or_update() {
    local name=$1
    local url=$2
    local pin=$3
    if [ -d "$name/.git" ]; then
        echo "[install] $name: already cloned, fetching..."
        git -C "$name" fetch --quiet
    else
        echo "[install] $name: cloning $url"
        git clone --quiet "$url" "$name"
    fi
    if [ -n "$pin" ]; then
        git -C "$name" checkout --quiet "$pin" || {
            echo "[install] WARN: could not check out $pin in $name; staying on default branch"
        }
    fi
}

# Pinned commits.  Replace 'main' with a specific SHA once the upstream repos
# tag a stable release; otherwise the pin floats with the default branch.
clone_or_update NSL                         https://github.com/thuml/Neural-Solver-Library.git    main
clone_or_update Transolver                  https://github.com/thuml/Transolver.git               main
clone_or_update FactFormer                  https://github.com/BaratiLab/FactFormer.git           main
clone_or_update gnot                        https://github.com/HaoZhongkai/GNOT.git               main
clone_or_update OFormer                     https://github.com/BaratiLab/OFormer.git              main
clone_or_update KoopmanLab                  https://github.com/Koopman-Laboratory/KoopmanLab.git  main
clone_or_update Orthogonal-Neural-operator  https://github.com/zwei-lin/ONO.git                   main
clone_or_update CNO                         https://github.com/bogdanraonic3/ConvolutionalNeuralOperator.git  main

cat <<EOF

[install] all baseline repositories cloned under $THIRD/.

Next steps:
  1. Inspect each repo's LICENSE before redistributing or modifying.
  2. From the release root, run a baseline trainer, e.g.:
       python baselines/ns_nsl.py --model_name FNO \\
              --data_path ./data/NavierStokes_V1e-5_N1200_T20.mat \\
              --baseline_root ./third_party/NSL
  3. See baselines/README.md for per-model invocation notes and known caveats.
EOF
