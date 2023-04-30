# Copyright (c) Sony AI Inc.
# All rights reserved.

import argparse

from avfs.dataset.data import AVFSDataset
from avfs.modeling.utils import (
    load_config,
    setup_environment,
    setup_experiment,
    setup_model,
    setup_optimizer,
    save_config_to_yaml,
)
from avfs.modeling.training import AVFSTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_filepath",
        required=True,
        type=str,
        help="The path to a training configuration .yaml file.",
    )
    args = parser.parse_args()

    cfg = load_config(cfg_filepath=args.cfg_filepath)

    setup_environment(
        devices=cfg["setup_cfg"]["devices"],
        primary_device=cfg["setup_cfg"]["primary_device"],
        numpy_seed=cfg["setup_cfg"]["numpy_seed"],
        pytorch_seed=cfg["setup_cfg"]["pytorch_seed"],
        random_seed=cfg["setup_cfg"]["random_seed"],
        cudnn_benchmark=cfg["setup_cfg"]["cudnn_benchmark"],
    )

    logger, cfg = setup_experiment(cfg=cfg)

    avfs_dataset = AVFSDataset(
        loader_cfg=cfg["loader_cfg"],
        mask_type=cfg["data_cfg"]["mask_type"],
        random_seed=cfg["setup_cfg"]["random_seed"],
        max_train_subset_prop=cfg["data_cfg"]["max_train_subset_prop"],
        validation_fivecrop_trans=False,
    )

    loaders, num_annotators, annotator_labels = avfs_dataset()
    cfg["misc_cfg"]["num_train_batches_to_average"] = min(
        cfg["misc_cfg"]["num_train_batches_to_average"], len(loaders["train"])
    )

    model, num_output_dims = setup_model(
        num_output_dims=cfg["model_cfg"]["num_output_dims"],
        data_parallel=cfg["setup_cfg"]["data_parallel"],
        devices=cfg["setup_cfg"]["devices"],
        primary_device=cfg["setup_cfg"]["primary_device"],
        create_masks=cfg["data_cfg"]["create_masks"],
        num_masks=num_annotators,
        chkpt_path=cfg["model_cfg"]["chkpt_path"],
        chkpt_filename=cfg["model_cfg"]["chkpt_filename"],
        freeze_encoder_gradients=cfg["model_cfg"]["freeze_encoder_gradients"],
        freeze_mask_gradients=cfg["model_cfg"]["freeze_mask_gradients"],
        posthoc_dims=cfg["model_cfg"]["posthoc_dims"],
        model_strict_load=cfg["misc_cfg"]["model_strict_load"],
    )

    cfg["model_cfg"]["num_output_dims"] = num_output_dims

    optimizer, start_epoch = setup_optimizer(
        model=model,
        optim_cfg=cfg["optim_cfg"],
        resume=cfg["misc_cfg"]["resume"],
        primary_device=cfg["setup_cfg"]["primary_device"],
        chkpt_path=cfg["model_cfg"]["chkpt_path"],
        chkpt_filename=cfg["model_cfg"]["chkpt_filename"],
    )

    save_config_to_yaml(cfg=cfg)

    trainer = AVFSTrainer(
        logger=logger,
        model=model,
        optimizer=optimizer,
        primary_device=cfg["setup_cfg"]["primary_device"],
        num_train_batches_to_average=cfg["misc_cfg"]["num_train_batches_to_average"],
        num_epochs=cfg["misc_cfg"]["num_epochs"],
        save_every_num_epochs=cfg["misc_cfg"]["save_every_num_epochs"],
        num_output_dims=cfg["model_cfg"]["num_output_dims"],
        learning_rate_cfg=cfg["learning_rate_cfg"],
        loss_cfg=cfg["loss_cfg"],
        monitor_validation_loss=cfg["misc_cfg"]["monitor_validation_loss"],
        annotator_labels=annotator_labels,
        debug=cfg["misc_cfg"]["debug"],
    )

    for epoch_i in range(start_epoch, cfg["misc_cfg"]["num_epochs"]):
        trainer.train(epoch_i=epoch_i, data_loader=loaders["train"], train_mode=True)

        trainer.validate(
            epoch_i=epoch_i, data_loader=loaders["validation"], train_mode=False
        )

        trainer.save_best_periodic(
            epoch_i=epoch_i, checkpoint_path=cfg["paths"]["checkpoint_path"]
        )


if __name__ == "__main__":
    main()
