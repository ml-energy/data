#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format mlenergy_data mlenergy tests
else
  ruff format --check mlenergy_data mlenergy tests
fi

ruff check mlenergy_data mlenergy tests
ty check mlenergy_data mlenergy tests
