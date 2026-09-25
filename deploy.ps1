# Deploy CTAdipo to shinyapps.io.
#
# Use this rather than calling rsconnect directly. It does two things that are easy to forget and
# that both silently inflate the upload:
#
#   1. Deletes __pycache__ and *.pyc. Running the test suite or importing the app regenerates them,
#      so they reappear between a manual clean and the deploy.
#   2. Excludes .git/ with -x globs. rsconnect-python has NO working ignore file -- .rscignore is an
#      R-only feature it does not read -- so exclusions must be passed on the command line. Without
#      this the repository history ships with the app: the first deploy after `git init` went out as
#      36 MB / 98 files instead of 19 MB / 31.
#
# The account is already registered on this machine, so no credentials appear here or are needed.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Get-ChildItem -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -File -Filter *.pyc -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

$bytes = (Get-ChildItem -Recurse -File |
          Where-Object { $_.FullName -notmatch '\\\.git\\' } |
          Measure-Object -Property Length -Sum).Sum
"bundle to upload: {0:N1} MB" -f ($bytes / 1MB)

rsconnect deploy shiny . `
    -n fullerlabtools `
    --title CTadipo `
    -x '.git/**' `
    -x '__pycache__/**' `
    -x '*.pyc' `
    -x 'rsconnect-python/**' `
    -x '_tta/**'
