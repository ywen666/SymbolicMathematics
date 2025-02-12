# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
import re
import sys
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset import ODEEgraphDataset
from .optim import get_optimizer
from .utils import to_cuda
from .envs import EnvDataset

if torch.cuda.is_available():
    import apex


logger = getLogger()


class Trainer(object):

    EQUATIONS = {}

    def __init__(self, modules, env, params):
        """
        Initialize trainer.
        """
        # modules / params
        self.modules = modules
        self.params = params
        self.contra_coeff = params.contra_coeff
        self.is_contrastive = (params.contra_coeff > 0)
        self.env = env

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = nn.parallel.DistributedDataParallel(self.modules[k], device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True)

        # set optimizers
        self.set_optimizers()

        # float16 / distributed (AMP)
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                for k in self.modules.keys():
                    self.modules[k] = apex.parallel.DistributedDataParallel(self.modules[k], delay_allreduce=True)

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [('processed_e', 0)] +
            [('processed_w', 0)] +
            sum([
                [(x, []), (f'{x}-AVG-STOP-PROBS', [])]
                for x in env.TRAINING_TASKS
            ], [])
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # file handler to export data
        if params.export_data:
            assert params.reload_data == ''
            params.export_path_prefix = os.path.join(params.dump_path, 'data.prefix')
            params.export_path_infix = os.path.join(params.dump_path, 'data.infix')
            self.file_handler_prefix = io.open(params.export_path_prefix, mode='a', encoding='utf-8')
            self.file_handler_infix = io.open(params.export_path_infix, mode='a', encoding='utf-8')
            logger.info(f"Data will be stored in prefix in: {params.export_path_prefix} ...")
            logger.info(f"Data will be stored in infix in: {params.export_path_infix} ...")

        # reload exported data
        if params.reload_data != '':
            assert params.export_data is False
            s = [x.split(',') for x in params.reload_data.split(';') if len(x) > 0]
            assert len(s) >= 1 and all(len(x) == 4 for x in s) and len(s) == len(set([x[0] for x in s]))
            self.data_path = {task: (train_path, valid_path, test_path) for task, train_path, valid_path, test_path in s}
            assert all(all(os.path.isfile(path) for path in paths) for paths in self.data_path.values())
            for task in self.env.TRAINING_TASKS:
                assert (task in self.data_path) == (task in params.tasks)
        else:
            self.data_path = None

        # create data loaders
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = {
                task: iter(self.env.create_train_iterator(task, params, self.data_path))
                for task in params.tasks
            }

            if self.is_contrastive:
                task = params.tasks[0]
                ode_dataset = EnvDataset(
                    env,
                    task=task,
                    train=True,
                    rng=None,
                    params=params,
                    path=self.data_path[task][0],
                )
                start_time = time.time()
                self.train_dataset = ODEEgraphDataset(ode_dataset, env)
                temp = time.time() - start_time
                print(f'initializing dataset takes {temp} secs.')
                train_sampler = RandomSampler(self.train_dataset)

                self.contrastive_dataloader = {
                    task: DataLoader(
                        self.train_dataset,
                        batch_size=params.batch_size,
                        shuffle=False,
                        sampler=train_sampler,
                        collate_fn=self.train_dataset.collate_fn,
                        num_workers=20,
                        pin_memory=True,
                    )
                }
                #self.contrastive_dataloader_iter = {
                #    task: iter(loader)
                #    for task, loader in self.contrastive_dataloader.items()
                #}
                #self.train_dataset.test_get(315)
                #start_time = time.time()
                #for _ in range(20):
                #    kk = self.contrastive_dataloader['ode1'].next()
                #temp = time.time() - start_time
                #print(f'20 dataloader iterations takes {temp} secs.')
                #import pdb; pdb.set_trace()

    def reinit_contrastive_dataloader(self):
        self.contrastive_dataloader_iter = {
            task: iter(loader)
            for task, loader in self.contrastive_dataloader.items()
        }

    def check_floats(self):
        self.success_list = []
        self.failed_list = []
        import time
        start_time = time.time()
        for idx in range(len(self.train_dataset)):
            if idx % 5000 == 0:
                print(f'============{idx}=============')
                current_time = time.time() - start_time
                print(f'Takes {current_time} secs')
            try:
                self.success_list.append(self.train_dataset[idx])
            except Exception as e:
                self.failed_list.append(idx)

    def get_train_sampler(self):
        if self.params.world_size <= 1:
            return RandomSampler(self.train_dataset)
        else:
            # Not sure if I should use gloabl rank here.
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.params.world_size,
                rank=self.params.global_rank,
                seed=self.params.env_base_seed,
            )

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters['model'] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert params.amp == 0 and params.fp16 is False or params.amp in [1, 2, 3] and params.fp16 is True
        mod_names = sorted(self.modules.keys())
        opt_names = sorted(self.optimizers.keys())
        modules, optimizers = apex.amp.initialize(
            [self.modules[k] for k in mod_names],
            [self.optimizers[k] for k in opt_names],
            opt_level=('O%i' % params.amp)
        )
        self.modules = {k: module for k, module in zip(mod_names, modules)}
        self.optimizers = {k: optimizer for k, optimizer in zip(opt_names, optimizers)}

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        if params.amp == -1:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                for name in names:
                    clip_grad_norm_(self.parameters[name], params.clip_grad_norm)
            for optimizer in optimizers:
                optimizer.step()

        # AMP optimization
        else:
            if self.n_iter % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizers) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    for name in names:
                        clip_grad_norm_(apex.amp.master_params(self.optimizers[name]), params.clip_grad_norm)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizers, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 20 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k.upper().replace('_', '-'), np.mean(v))
            for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = ""
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - ".format(
            self.stats['processed_e'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_e'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
            'params': {k: v for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning(f"Saving {name} optimizer ...")
                data[f'{name}_optimizer'] = self.optimizers[name].state_dict()

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        print(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        for k, v in self.modules.items():
            v.load_state_dict(data[k])

        # reload optimizers
        for name in self.optimizers.keys():
            # AMP checkpoint reloading is buggy, we cannot reload optimizers
            # instead, we only reload current iterations / learning rates
            if self.params.amp == -1:
                logger.warning(f"Reloading checkpoint optimizer {name} ...")
                self.optimizers[name].load_state_dict(data[f'{name}_optimizer'])
            else:
                logger.warning(f"Not reloading checkpoint optimizer {name}.")
                for group_id, param_group in enumerate(self.optimizers[name].param_groups):
                    if 'num_updates' not in param_group:
                        logger.warning(f"No 'num_updates' for optimizer {name}.")
                        continue
                    logger.warning(f"Reloading 'num_updates' and 'lr' for optimizer {name}.")
                    param_group['num_updates'] = data[f'{name}_optimizer']['param_groups'][group_id]['num_updates']
                    param_group['lr'] = self.optimizers[name].get_lr_for_step(param_group['num_updates'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint('checkpoint')
        self.epoch += 1

    def get_batch(self, task):
        """
        Return a training batch for a specific task.
        """
        try:
            if self.is_contrastive:
                batch = next(self.contrastive_dataloader_iter[task])
            else:
                batch = next(self.dataloader[task])
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} when fetching batch. "
                         "Arguments:{2!r}. Restarting ...".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, e.args))
            if self.params.is_slurm_job:
                if int(os.environ['SLURM_PROCID']) == 0:
                    logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
                    os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
                else:
                    logger.warning("Not the master process, no need to requeue.")
            raise

        return batch

    def export_data(self, task):
        """
        Export data to the disk.
        """
        env = self.env
        (x1, len1), (x2, len2), _ = self.get_batch(task)
        for i in range(len(len1)):
            # prefix
            prefix1 = [env.id2word[wid] for wid in x1[1:len1[i] - 1, i].tolist()]
            prefix2 = [env.id2word[wid] for wid in x2[1:len2[i] - 1, i].tolist()]
            # infix
            infix1 = env.prefix_to_infix(env.unclean_prefix(prefix1))[1:-1]
            infix2 = env.prefix_to_infix(env.unclean_prefix(prefix2))[1:-1]
            # save
            prefix1_str = ' '.join(prefix1)
            prefix2_str = ' '.join(prefix2)
            infix1_str = infix1.replace('f(x)', 'y').replace('Derivative(y,x)', "y'").replace("Derivative(y',x)", "y''")
            infix2_str = infix2.replace('f(x)', 'y').replace('Derivative(y,x)', "y'").replace("Derivative(y',x)", "y''")
            infix1_str = re.sub(r'(?<![a-zA-Z0-9])\(([a-zA-Z0-9]+)\)(?<![a-zA-Z0-9])', r'\1', infix1_str)
            infix2_str = re.sub(r'(?<![a-zA-Z0-9])\(([a-zA-Z0-9]+)\)(?<![a-zA-Z0-9])', r'\1', infix2_str)
            self.file_handler_prefix.write(f'{prefix1_str}\t{prefix2_str}\n')
            self.file_handler_infix.write(f'{infix1_str}\t{infix2_str}\n')
            self.file_handler_prefix.flush()
            self.file_handler_infix.flush()
            self.EQUATIONS[(prefix1_str, prefix2_str)] = self.EQUATIONS.get((prefix1_str, prefix2_str), 0) + 1

        # number of processed sequences / words
        self.n_equations += self.params.batch_size
        self.stats['processed_e'] += len1.size(0)
        self.stats['processed_w'] += (len1 + len2 - 2).sum().item()

    def enc_dec_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.train()
        decoder.train()

        # batch
        if self.is_contrastive:
            (x1, len1), (x2, len2), (sampled_x1, sampled_len), _ = self.get_batch(task)
        else:
            (x1, len1), (x2, len2), _ = self.get_batch(task)
        #(x1, len1), (x2, len2), _ = self.get_batch(task)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        if self.is_contrastive:
            x1, len1, x2, len2, y, sampled_x1, sampled_len = to_cuda(
                x1, len1, x2, len2, y, sampled_x1, sampled_len)
        else:
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
        #x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
        #print(f'x1 shape is {x1.shape}')
        #print(f'x1 len shape is {len1.shape}')
        #print(f'x2 shape is {x2.shape}')
        #print(f'x2 len shape is {len2.shape}')
        #print(f'sampled x1 shape is {sampled_x1.shape}')
        #print(f'sampled len shape is {sampled_len.shape}')

        # forward / loss
        encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
        decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)

        _, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)
        if self.is_contrastive:
            sampled_encoded = encoder(
                'fwd', x=sampled_x1, lengths=sampled_len, causal=False)
            sampled_decoded = decoder(
                'fwd', x=x2, lengths=len2, causal=True,
                src_enc=sampled_encoded.transpose(0, 1), src_len=sampled_len)
            _, sampled_decoder_loss = decoder(
                'predict', tensor=sampled_decoded, pred_mask=pred_mask,
                y=y, get_scores=False)

            contrastive_states = torch.cat(
                [encoded[len1 - 1, torch.arange(encoded.size(1))],
                sampled_encoded[len2 - 1, torch.arange(sampled_encoded.size(1))]],
                dim=0
            )
            contra_logits, contra_labels = self.info_nce_loss(contrastive_states)
            contra_ce = nn.CrossEntropyLoss()
            contra_loss = contra_ce(contra_logits, contra_labels)
            loss += sampled_decoder_loss + self.contra_coeff * contra_loss

        self.stats[task].append(loss.item())

        # optimize
        self.optimize(loss)

        # number of processed sequences / words
        self.n_equations += params.batch_size
        self.stats['processed_e'] += len1.size(0)
        self.stats['processed_w'] += (len1 + len2 - 2).sum().item()

    def info_nce_loss(self, features, n_views=2, temperature=0.5):
        # features is of shape [2, |kwargs['contrastive_nodes']|, hidden]
        batch_size = features.size(0) // n_views
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / temperature
        return logits, labels
