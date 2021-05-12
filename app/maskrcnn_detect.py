import numpy as np
import app.maskrcnn_utils as rcnn_utils
import app.maskrcnn_config as rcnn_config

config = rcnn_config.Config()


def mold_inputs(images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matrices [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matrices:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        # Resize image
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding, crop = rcnn_utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        molded_image = rcnn_utils.mold_image(molded_image, config)
        # Build image_meta
        image_meta = rcnn_utils.compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows


def get_anchors(image_shape):
    """Returns anchor pyramid for the given image size."""
    backbone_shapes = rcnn_utils.compute_backbone_shapes(config, image_shape)
    # Cache anchors and reuse if image shape is the same
    # if not hasattr("_anchor_cache"):
    _anchor_cache = {}
    if not tuple(image_shape) in _anchor_cache:
        # Generate Anchors
        a = rcnn_utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)
        # Keep a copy of the latest anchors in pixel coordinates because
        # it's used in inspect_model notebooks.
        # TODO: Remove this after the notebook are refactored to not use it
        anchors = a
        # Normalize coordinates
        _anchor_cache[tuple(image_shape)] = rcnn_utils.norm_boxes(a, image_shape[:2])
    return _anchor_cache[tuple(image_shape)]


def detect(images):
    """Runs the detection pipeline.

    images: List of images, potentially of different sizes.

    Returns a list of dicts, one dict per image. The dict contains:
    rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    class_ids: [N] int class IDs
    scores: [N] float probability scores for the class IDs
    masks: [H, W, N] instance binary masks
    """

    # assert mode == "inference", "Create model in inference mode."
    assert len(images) == config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

    # Mold inputs to format expected by the neural network
    molded_images, image_metas, windows = mold_inputs(images)

    # Validate image sizes
    # All images in a batch MUST be of the same size
    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

    # Anchors
    anchors = get_anchors(image_shape)
    # Duplicate across the batch dimension because Keras requires it
    # TODO: can this be optimized to avoid duplicating the anchors?
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

    # # Run object detection
    # detections, _, _, mrcnn_mask, _, _, _ = \
    #     self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
    # # Process detections
    # results = []
    # for i, image in enumerate(images):
    #     final_rois, final_class_ids, final_scores, final_masks = \
    #         self.unmold_detections(detections[i], mrcnn_mask[i],
    #                                image.shape, molded_images[i].shape,
    #                                windows[i])
    #     results.append({
    #         "rois": final_rois,
    #         "class_ids": final_class_ids,
    #         "scores": final_scores,
    #         "masks": final_masks,
    #     })
    return molded_images, image_metas, anchors, windows
