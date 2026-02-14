#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format mlenergy_data tests
else
  ruff format --check mlenergy_data tests
fi

ruff check mlenergy_data tests
ty check mlenergy_data tests
