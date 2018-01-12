#!/bin/bash
export RUN_DIR=$(dirname $( readlink -f ${0} ) )
source ${RUN_DIR}/../../export-data/LibAzaleadTool/run_db.env

FLASK_APP=serverpredictwithoutserving.py flask run --host=94.23.54.195

exit $FLASK_APP
