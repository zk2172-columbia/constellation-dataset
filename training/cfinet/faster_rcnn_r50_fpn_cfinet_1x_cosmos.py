_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

find_unused_parameters=True
rpn_weight = 0.9
model = dict(
    type='FasterRCNN',
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    rpn_head=dict(
        _delete_=True,
        type='CRPNHead',
        num_stages=2,
        stages=[
            dict(
                type='StageRefineRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[2],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32]),
                refine_reg_factor=200.0,
                refine_cfg=dict(type='dilation', dilation=3),
                refined_feature=True,
                sampling=False,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageRefineRPNHead',
                in_channels=256,
                feat_channels=256,
                refine_cfg=dict(type='offset'),
                refined_feature=True,
                sampling=True,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))]),
    roi_head=dict(
        type='FIRoIHead',
        num_gpus=1,
        temperature=0.6,
        contrast_loss_weights=0.50,
        num_con_queue=256,
        num_classes=2,
        con_sampler_cfg=dict(
            num=128,
            pos_fraction=[0.5, 0.25, 0.125]),
        con_queue_dir="./work_dirs/roi_feats/cfinet",
        ins_quality_assess_cfg=dict(
            cls_score=0.05,
            hq_score=0.65,
            lq_score=0.25,
            hq_pro_counts_thr=8),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
# model training and testing settings
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='DynamicAssigner',
                    low_quality_iou_thr=0.2,
                    base_pos_iou_thr=0.25,
                    neg_iou_thr=0.15),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        rpn_proposal=dict(max_per_img=300, nms=dict(iou_threshold=0.8)),
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.50, neg_iou_thr=0.50, min_pos_iou=0.50),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5))),
    test_cfg=dict(
        rpn=dict(max_per_img=300, nms=dict(iou_threshold=0.5)),
        rcnn=dict(score_thr=0.05))
)

fp16 = dict(loss_scale='dynamic')   # mixed precision

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={
                     'roi_head.fc_enc': dict(lr_mult=0.05), 
                     'roi_head.fc_proj': dict(lr_mult=0.05)})
                 )

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 125
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=125)
evaluation = dict(interval=1, classwise=True, metric='bbox')
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=5, out_dir='cfinet_nopretrain_constellation_weights')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs={'project': 'constellation'},
            interval=50,
            log_artifact=True
            )
    ])

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('vehicle', 'pedestrian',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=22,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        type=dataset_type,
        img_prefix='/constellation/images/train',
        classes=classes,
        ann_file='/constellation/train_coco.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix='/constellation/images/val',
        classes=classes,
        ann_file='/constellation/val_coco.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix='/constellation/images/val',
        classes=classes,
        ann_file='/constellation/val_coco.json',
        pipeline=test_pipeline))
# We can use a pre-trained Mask RCNN model. (Refer to official CFINet repo for weights)
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
