"""
This file is expecting the 'pipeline' package name to be defined
This file completely defines the experiment to run
"""
import os
from random import random
from typing import Tuple, Generator, Optional, Dict, List

import numpy as np
import torch
import torch.utils.data

from pipeline import data
from pipeline import logger
from pipeline.discriminators import CaloganPhysicsDiscriminator3D
from pipeline.evaluation import evaluate_model
from pipeline.experiment_setup import experiments_storage, global_config, init_logger
from pipeline.gan import GAN
from pipeline.generators import CaloganPhysicsGenerator3D
from pipeline.fhcal_virtual25_gnn import VirtualCNNGenerator, GraphCritic
from pipeline.fhcal_data import load_geometry, PhysicalModulesDataset
from pipeline.metrics import *
from pipeline.config import load_global_config
from pipeline.custom_metrics import *
from pipeline.predicates import TrainPredicate, IgnoreFirstNEpochsPredicate, EachNthEpochPredicate
from pipeline.regularizer import *
from pipeline.results_storage import ResultsStorage
from pipeline.storage import ExperimentsStorage
from pipeline.train import Stepper, WganEpochTrainer, GanTrainer


def form_metric() -> Metric:
    return MetricsSequence(
        # BetaMetric(),
        # DiscriminatorParameterMetric('r'),
        CriticValuesDistributionMetric(values_cnt=1000),
        # Grid-based PhysicsDataStatistics expects [B,L,7,9].
        # Keep the existing critic metric for physical-module tensors.
    )


def form_metric_predicate() -> Optional[TrainPredicate]:
    return IgnoreFirstNEpochsPredicate(0) & EachNthEpochPredicate(5)
    # return None


def form_result_metrics() -> Metric:
    return MetricsSequence(
                DataStatistics(
                    KLDivergence(LongitudualClusterAsymmetryMetric()),
                    KLDivergence(TransverseClusterAsymmetryMetric()),
                    KLDivergence(ClusterLongitudualWidthMetric()),
                    KLDivergence(ClusterTransverseWidthMetric()),
                ),
            )


def form_dataset(train: bool = False) -> torch.utils.data.Dataset:
    data_filepath = global_config.paths.data_dir_path + '/fhcal_data_muons_81619.npz'

    assert os.path.exists(data_filepath), (
        f'Dataset not found: {data_filepath}'
    )

    geometry = load_geometry(os.path.join(global_config.paths.data_dir_path, 'fhcal_geometry.json'))
    dataset = PhysicalModulesDataset(
        data_filepath, geometry,
        module_map=os.path.join(global_config.paths.data_dir_path, 'fhcal_module_map.json'),
        train=train,
    )
    return data.UnifiedDatasetWrapper(dataset)


def form_gan_trainer(model_name: str, gan_model: Optional[GAN] = None, n_epochs: int = 100) -> Generator[Tuple[int, GAN], None, GAN]:
    """
    :return: a generator that yields (epoch number, gan_model after this epoch)
    """
    logger_cm_fn = init_logger(model_name)
    metric = form_metric()
    metric_predicate = form_metric_predicate()

    train_dataset = form_dataset(train=True)
    val_dataset = form_dataset(train=False)

    # for local testing
    val_size = int(0.1 * len(val_dataset))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(val_size))
    # -------
    noise_dimension = 64

    def normal_noise_generator(n: int) -> torch.Tensor:
        return torch.randn(n, noise_dimension)

    geometry = load_geometry(os.path.join(global_config.paths.data_dir_path, 'fhcal_geometry.json'))
    condition_dim = train_dataset.dataset.condition_dim
    generator = VirtualCNNGenerator(
        geometry, condition_dim, noise_dim=noise_dimension,
        condition_state=train_dataset.dataset.condition_state,
    )
    discriminator = GraphCritic(geometry, condition_dim)
    # discriminator = apply_normalization(discriminator, MultiplyOutputNormalizer, coef=2., is_trainable_coef=False)
    # discriminator = apply_normalization(discriminator, WeakSpectralNormalizer, beta=2., is_trainable_beta=False)
    # discriminator = apply_normalization(discriminator, ABCASNormalizer)

    # regularizer = BasicRegularizer(lambda: sum(p.norm(2) for p in discriminator.parameters()))
    # regularizer = PowRegularizer(regularizer, 2)
    # regularizer = MultiplierRegularizer(regularizer, start_value=1.)
    regularizer = None
    # lambd = 1.
    # normalization_loss = lambda: lambd*discriminator.coef**2

    if gan_model is None:
        gan_model = GAN(generator, discriminator, normal_noise_generator)
    else:
        generator, discriminator = gan_model.generator, gan_model.discriminator


    generator_stepper = Stepper(
      optimizer=torch.optim.Adam(
          generator.parameters(),
          lr=1e-4, betas=(0.0, 0.9)
      )
    )


    discriminator_stepper = Stepper(
        optimizer=torch.optim.Adam(
            discriminator.parameters(),
            lr=1e-4, betas=(0.0, 0.9)
        )
    )

    epoch_trainer = WganEpochTrainer(
        n_critic=5,
        batch_size=100,
        physical_modules=True,
        lambda_gp=10.0,
        lambda_layer_fraction=0.0,
        lambda_quantile=0.0,
        lambda_peak=0.0,
        lambda_layer_distribution=0.0,
        lambda_conditional=0.0
    )

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=True, save_checkpoint_once_in_epoch=5)
    train_gan_generator = trainer.train(gan_model=gan_model,
                                        train_dataset=train_dataset, val_dataset=val_dataset,
                                        generator_stepper=generator_stepper,
                                        critic_stepper=discriminator_stepper,
                                        epoch_trainer=epoch_trainer,
                                        n_epochs=n_epochs,
                                        metric=metric, metric_predicate=metric_predicate,
                                        logger_cm_fn=logger_cm_fn,
                                        regularizer=regularizer)
    return train_gan_generator, epoch_trainer


def run() -> GAN:
    model_name = 'physics_virtual25_gnn'
    gan_trainer, epoch_trainer = form_gan_trainer(model_name=model_name, n_epochs=15)
    gan = None
    for epoch, gan in gan_trainer:
        pass
    # evaluate model somehow ...
    return gan


if __name__ == '__main__':
    run()

