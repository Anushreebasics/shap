from importlib import import_module
from typing import Any

import lazy_loader as lazy

from ._explanation import Cohorts as Cohorts
from ._explanation import Explanation as Explanation

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

try:
    # Version from setuptools-scm
    from ._version import version as __version__
except ImportError:
    # Expected when running locally without build
    __version__ = "0.0.0-not-built"

_no_matplotlib_warning = (
    "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
)

_PLOT_EXPORTS = {
    "plots",
    "bar_plot",
    "summary_plot",
    "decision_plot",
    "multioutput_decision_plot",
    "embedding_plot",
    "force_plot",
    "getjs",
    "initjs",
    "save_html",
    "group_difference_plot",
    "heatmap_plot",
    "image_plot",
    "monitoring_plot",
    "partial_dependence_plot",
    "dependence_plot",
    "text_plot",
    "violin_plot",
    "waterfall_plot",
}

_PLOT_ALIAS_MAP = {
    "bar_plot": ("shap.plots._bar", "bar_legacy"),
    "summary_plot": ("shap.plots._beeswarm", "summary_legacy"),
    "decision_plot": ("shap.plots._decision", "decision"),
    "multioutput_decision_plot": ("shap.plots._decision", "multioutput_decision"),
    "embedding_plot": ("shap.plots._embedding", "embedding"),
    "force_plot": ("shap.plots._force", "force"),
    "group_difference_plot": ("shap.plots._group_difference", "group_difference"),
    "heatmap_plot": ("shap.plots._heatmap", "heatmap"),
    "image_plot": ("shap.plots._image", "image"),
    "monitoring_plot": ("shap.plots._monitoring", "monitoring"),
    "partial_dependence_plot": ("shap.plots._partial_dependence", "partial_dependence"),
    "dependence_plot": ("shap.plots._scatter", "dependence_legacy"),
    "text_plot": ("shap.plots._text", "text"),
    "violin_plot": ("shap.plots._violin", "violin"),
    "waterfall_plot": ("shap.plots._waterfall", "waterfall"),
}

_lazy_getattr = __getattr__


def __getattr__(name: str) -> Any:
    if name in _PLOT_EXPORTS:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise ImportError(_no_matplotlib_warning) from exc
    if name in _PLOT_ALIAS_MAP:
        module_name, attr_name = _PLOT_ALIAS_MAP[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    return _lazy_getattr(name)
