import os
from utils.file._yaml import load_yaml

import os
from utils.file._yaml import load_yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENV = os.path.join(BASE_DIR, "uniform_env.yaml")
ENV = load_yaml(ENV)

GAUGE_FIELDS = os.path.join(BASE_DIR, "gauge_fields.yaml")
GAUGE_FIELDS = load_yaml(GAUGE_FIELDS)
"""GAUGE_FIELDS = {
    j:float(l)
    for k, v in GAUGE_FIELDS.values()
    for j, l in v.values()
}"""


QUARKS = os.path.join(BASE_DIR, "quarks.yaml")
QUARKS = load_yaml(QUARKS)
"""QUARKS = {
    j:float(l)
    for k, v in QUARKS.values()
    for j, l in v.values()
}"""


QF_LEX=os.path.join(BASE_DIR, "qf_lex.yaml")
QF_LEX=load_yaml(QF_LEX)

FERM_PARAMS = os.path.join(BASE_DIR, "psi_fields.yaml")
FERM_PARAMS = load_yaml(FERM_PARAMS)
"""FERM_PARAMS = {
    j:float(l)
    for k, v in FERM_PARAMS.values()
    for j, l in v.values()
}
"""
