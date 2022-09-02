import yaml
import json
import numbers
import numpy as np

# Copied from ray.tune.logger:
# https://github.com/ray-project/ray/blob/04bfba12742841bc0112d645ce85c537903fdf99/python/ray/tune/logger.py#L722
# and ray.tune.utils.util
# https://github.com/ray-project/ray/blob/04bfba12742841bc0112d645ce85c537903fdf99/python/ray/tune/utils/util.py#L835

class SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__
                    and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(SafeFallbackEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)

def pretty_print(result):
    # Copied from ray.tune.logger:
    # https://github.com/ray-project/ray/blob/04bfba12742841bc0112d645ce85c537903fdf99/python/ray/tune/logger.py#L722

    result = result.copy()
    out = {} # Clear None valued keys
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)