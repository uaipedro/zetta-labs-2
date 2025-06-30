"""
Componentes do Dashboard
"""

from .spatial_analysis import render_spatial_analysis
from .temporal_analysis import render_temporal_analysis
from .model_analysis import render_model_analysis

__all__ = [
    'render_executive_summary',
    'render_spatial_analysis', 
    'render_temporal_analysis',
    'render_model_analysis'
] 