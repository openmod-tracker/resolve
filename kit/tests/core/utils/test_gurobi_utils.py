from new_modeling_toolkit.core.utils.gurobi_utils import _GUROBI_API_ACCESS_ID_HEADER_PARAM_NAME
from new_modeling_toolkit.core.utils.gurobi_utils import _GUROBI_API_BASE_URL
from new_modeling_toolkit.core.utils.gurobi_utils import _GUROBI_API_SECRET_KEY_HEADER_PARAM_NAME
from new_modeling_toolkit.core.utils.gurobi_utils import _GUROBI_LICENSE_CLOUDACCESSID_VARIABLE_NAME
from new_modeling_toolkit.core.utils.gurobi_utils import _GUROBI_LICENSE_CLOUDKEY_VARIABLE_NAME
from new_modeling_toolkit.core.utils.gurobi_utils import _GUROBI_LICENSE_CLOUDPOOL_VARIABLE_NAME
from new_modeling_toolkit.core.utils.gurobi_utils import _GUROBI_LICENSE_LICENSEID_VARIABLE_NAME


def test_constants():
    assert _GUROBI_API_BASE_URL == "https://cloud.gurobi.com/api/v2"
    assert _GUROBI_API_ACCESS_ID_HEADER_PARAM_NAME == "X-GUROBI-ACCESS-ID"  # nosec
    assert _GUROBI_API_SECRET_KEY_HEADER_PARAM_NAME == "X-GUROBI-SECRET-KEY"  # nosec

    assert _GUROBI_LICENSE_CLOUDACCESSID_VARIABLE_NAME == "CLOUDACCESSID"
    assert _GUROBI_LICENSE_CLOUDKEY_VARIABLE_NAME == "CLOUDKEY"
    assert _GUROBI_LICENSE_LICENSEID_VARIABLE_NAME == "LICENSEID"
    assert _GUROBI_LICENSE_CLOUDPOOL_VARIABLE_NAME == "CLOUDPOOL"
