"""
The MIT YOLO rewrite
"""

__autogen__ = """
mkinit ~/code/YOLO-v9/yolo/__init__.py --nomods --write --lazy-loader

# Check to see how long it takes to run a simple help command
time python -m yolo.lazy --help
"""

__submodules__ = {
    'config.config': ['Config', 'NMSConfig'],
    'model.yolo': ['create_model'],
    'tools.data_loader': ['AugmentationComposer', 'create_dataloader'],
    'tools.drawer': ['draw_bboxes'],
    'tools.solver': ['TrainModel'],
    'utils.bounding_box_utils': ['Anc2Box', 'Vec2Box', 'bbox_nms', 'create_converter'],
    'utils.deploy_utils': ['FastModelLoader'],
    'utils.logging_utils': [
        'ImageLogger', 'YOLORichModelSummary',
        'YOLORichProgressBar',
        'validate_log_directory'
    ],
    'utils.model_utils': ['PostProcess'],
}


import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={},
    submod_attrs={
        'config.config': [
            'Config',
            'NMSConfig',
        ],
        'model.yolo': [
            'create_model',
        ],
        'tools.data_loader': [
            'AugmentationComposer',
            'create_dataloader',
        ],
        'tools.drawer': [
            'draw_bboxes',
        ],
        'tools.solver': [
            'TrainModel',
        ],
        'utils.bounding_box_utils': [
            'Anc2Box',
            'Vec2Box',
            'bbox_nms',
            'create_converter',
        ],
        'utils.deploy_utils': [
            'FastModelLoader',
        ],
        'utils.logging_utils': [
            'ImageLogger',
            'YOLORichModelSummary',
            'YOLORichProgressBar',
            'validate_log_directory',
        ],
        'utils.model_utils': [
            'PostProcess',
        ],
    },
)

__all__ = ['Anc2Box', 'AugmentationComposer', 'Config', 'FastModelLoader',
           'ImageLogger', 'NMSConfig', 'PostProcess', 'TrainModel', 'Vec2Box',
           'YOLORichModelSummary', 'YOLORichProgressBar', 'bbox_nms',
           'create_converter', 'create_dataloader', 'create_model',
           'draw_bboxes', 'validate_log_directory']
