import os

__PROJECT_NAME__ = "BERTAP_v2"
__PROJECT_ROOT__ = os.path.abspath(__file__).split(__PROJECT_NAME__)[0] + __PROJECT_NAME__

BASE_DIR = {
    "project": __PROJECT_ROOT__,
    "save": os.path.join(__PROJECT_ROOT__, 'save')
}