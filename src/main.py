from models.visual.visual_model import get_visual_model
from utils.globals import logger
from prepare_dataset.dataset import VideoDataset


datasets = [
    VideoDataset(
        dataset_name="XD-Violence",
        dataset_path="../data/XD-Violence",
        label_path='class.txt',
        batch_size=16
    )
]
# Iterate all datasets
for dataset in datasets:
    # pre-process datasets
    logger.debug(f'pre-processing dataset: {dataset.dataset_name}')
    dataset_violent = dataset.dataset_builder()

    rgb_model = get_visual_model(input_shape=(32, 224, 224, 3), num_classes=len(dataset_violent.dataset_labels))
    flow_model = get_visual_model(input_shape=(32, 224, 224, 2), num_classes=len(dataset_violent.dataset_labels),
                                  type_model='flow')

    for epoch in range(5):
        for video_index in range(dataset_violent.dataset_count):
            generator = dataset_violent.get_batch(video_index)
            for (frame_batch, flow_batch, label_batch) in generator:
                rgb_model.fit(x=frame_batch, y=label_batch, batch_size=dataset_violent.batch_size, verbose=1)
                # flow_model.fit(x=flow_batch, y=label_batch, batch_size=dataset_violent.batch_size, verbose=1)


    # rgb_example = dataset_violent.videos_frames[0][None, ...]
    # flow_example = dataset_violent.videos_flows[0][None, ...]
    # #
    # rgb_logits = rgb_model.predict(rgb_example)
    # flow_logits = flow_model.predict(flow_example)
    # sample_logits = rgb_logits + flow_logits
    # #
    # # produce softmax output from model logit for class probabilities
    # sample_logits = sample_logits[0]  # we are dealing with just one example
    # sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))
    #
    # sorted_indices = np.argsort(sample_predictions)[::-1]
    # print('\nNorm of logits: %f' % np.linalg.norm(sample_logits))
    # print('\nTop 7 classes and probabilities')
    # for index in sorted_indices[:7]:
    #     print(sample_predictions[index], sample_logits[index], list(dataset_violent.dataset_labels.keys())[index])

    #  # Split dataset to test and train
    #     x_train, x_test, y_train, y_test = train_test_split(videos_frames_paths, videos_labels, test_size=0.20,
    #                                                         random_state=42)
    #
    # # Split dataset to test and validation
    # x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.20,
    #                                                                 random_state=42)
