# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for decoders."""

import logging
import math

import numpy as np
import os
import shutil

from neural_sp.models.base import ModelBase

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class DecoderBase(ModelBase):
    """Base class for decoders."""

    def __init__(self):

        super(ModelBase, self).__init__()

        logger.info('Overriding DecoderBase class.')

    def reset_session(self):
        self._new_session = True

    def trigger_scheduled_sampling(self):
        logger.info('Activate scheduled sampling')
        self._ss_prob = getattr(self, 'ss_prob', 0)

    def trigger_quantity_loss(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            logger.info('Activate quantity loss')
            self._quantity_loss_weight = getattr(self, 'quantity_loss_weight', 0)

    def trigger_latency_loss(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            logger.info('Activate latency loss')
            self._latency_loss_weight = getattr(self, 'latency_loss_weight', 0)

    def trigger_stableemit(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            if hasattr(self, 'score'):
                self.score.trigger_stableemit()
            elif hasattr(self, 'layers'):
                pass  # TODO(hirofumi): MMA

    def greedy(self, eouts, elens, max_len_ratio):
        raise NotImplementedError

    def embed_token_id(self, indices):
        raise NotImplementedError

    def cache_embedding(self, device):
        raise NotImplementedError

    def initialize_beam(self, hyp, lmstate):
        raise NotImplementedError

    def beam_search(self, eouts, elens, params, idx2token):
        raise NotImplementedError

    def _plot_attention(self, save_path=None, n_cols=2):
        """Plot attention for each head in all decoder layers."""
        if len(getattr(self, 'aws_dict', {}).keys()) == 0:
            return

        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        elens = self.data_dict['elens']
        ylens = self.data_dict['ylens']
        # ys = self.data_dict['ys']
        aws_dict = self.aws_dict

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        for k, aw in aws_dict.items():
            if aw is None:
                continue

            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols * max(1, n_heads // 4)
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp,
                                     figsize=(20 * max(1, n_heads // 4), 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                if 'yy' in k:
                    ax.imshow(aw[-1, h, :ylens[-1], :ylens[-1]], aspect="auto")
                else:
                    ax.imshow(aw[-1, h, :ylens[-1], :elens[-1]], aspect="auto")
                # NOTE: show the last utterance in a mini-batch
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # ax.set_yticks(np.linspace(0, ylens[-1] - 1, ylens[-1]))
                # ax.set_yticks(np.linspace(0, ylens[-1] - 1, 1), minor=True)
                # ax.set_yticklabels(ys + [''])

            fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, '%s.png' % k))
            plt.close()

    def _enhanced_plot_attention(self, save_path=None, batch_info=None, reporter=None, idx2token=None):
        """Plot attention for each head in all decoder layers."""
        if len(getattr(self, 'aws_dict', {}).keys()) == 0:
            return
        matplotlib.rc('font', size=8)
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['simhei']
        # 用来正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

        elens = self.data_dict['elens']
        ylens = self.data_dict['ylens']
        # ys = self.data_dict['ys']
        aws_dict = self.aws_dict

        for k, aw in aws_dict.items():
            if aw is None:
                continue

            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else math.floor(math.sqrt(n_heads))
            n_rows_tmp = max(1, n_heads // n_cols_tmp)
            for sample_idx, sample_id in enumerate(batch_info["utt_ids"]):
                plt.clf()
                fig, axes = plt.subplots(n_rows_tmp, n_cols_tmp, figsize=(4 * n_cols_tmp, 4 * n_rows_tmp), squeeze=False)
                txt = idx2token(self.data_dict['ys'][sample_idx][:ylens[sample_idx] - 1], return_list=True)
                txt.append("EOS")
                for h in range(n_heads):
                    ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                    if 'yy' in k:
                        ax.imshow(aw[sample_idx, h, :ylens[sample_idx], :ylens[sample_idx]], aspect="auto")
                        ax.set_yticks(range(len(txt)))
                        ax.set_xticks(range(len(txt)))
                        ax.set_xticklabels(txt)
                        ax.set_yticklabels(txt)
                        ax.set_xlabel(f"Input(head{h})")
                        ax.set_ylabel(f"Output(head{h})")
                    else:
                        ax.imshow(aw[-1, h, :ylens[sample_idx], :elens[sample_idx]], aspect="auto")
                        ax.set_yticks(range(len(txt)))
                        ax.set_yticklabels(txt)
                        ax.set_xlabel(f"Feature(head{h})")
                        ax.set_ylabel(f"Token(head{h})")
                    # NOTE: show the last utterance in a mini-batch
                    ax.grid(False)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                fig.tight_layout()
                reporter.add_figure(f"validate/dec_att_weights/{sample_id}/{k}", fig)
                plt.close()

    def _plot_ctc(self, save_path=None, topk=10):
        """Plot CTC posterior probabilities."""
        if self.ctc_weight == 0:
            return
        if len(self.ctc.prob_dict.keys()) == 0:
            return

        from matplotlib import pyplot as plt

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        elen = self.ctc.data_dict['elens'][-1]
        probs = self.ctc.prob_dict['probs'][-1, :elen]  # `[T, vocab]`
        # NOTE: show the last utterance in a mini-batch
        # 对于每个token的概率分布，排序
        topk_ids = np.argsort(probs, axis=1)

        plt.clf()
        n_frames = probs.shape[0]
        times_probs = np.arange(n_frames)
        plt.figure(figsize=(20, 8))

        # NOTE: index 0 is reserved for blank
        for idx in set(topk_ids.reshape(-1).tolist()):
            if idx == 0:
                plt.plot(times_probs, probs[:, 0], ':', label='<blank>', color='grey')
            else:
                plt.plot(times_probs, probs[:, idx])
        plt.xlabel(u'Time [frame]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xticks(list(range(0, int(n_frames) + 1, 10)))
        plt.yticks(list(range(0, 2, 1)))

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'prob.png'))
        plt.close()

    def _enhanced_plot_ctc(self, save_path=None, topk=10, batch_info=None, reporter=None, idx2token=None, tag="valid"):
        """Plot CTC posterior probabilities."""
        if self.ctc_weight == 0:
            return
        if len(self.ctc.prob_dict.keys()) == 0:
            return

        from matplotlib import pyplot as plt
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['simhei']
        # 设置xtick字体大小
        matplotlib.rcParams['xtick.labelsize'] = 10
        # 用来正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        for sample_idx, sample_id in enumerate(batch_info["utt_ids"]):
            elen = self.ctc.data_dict['elens'][sample_idx]
            probs = self.ctc.prob_dict['probs'][sample_idx, :elen]  # `[T, vocab]`
            # NOTE: show the last utterance in a mini-batch
            # 对于每个token的概率分布，排序
            topk_ids = np.argsort(probs, axis=1)
            topk_ids = topk_ids[:, -topk:]
            plt.clf()
            n_frames = probs.shape[0]
            times_probs = np.arange(n_frames)
            figure = plt.figure(figsize=(16, 2))
            # NOTE: index 0 is reserved for blank
            for idx in set(topk_ids.reshape(-1).tolist()):
                if idx == 0:
                    plt.plot(times_probs, probs[:, 0], ':', label='<blank>', color='grey')
                else:
                    plt.plot(times_probs, probs[:, idx])
            txt = idx2token(topk_ids[:, -1], return_list=True)
            plt.xticks(times_probs, txt, rotation=45)
            plt.xlabel(f"frames({batch_info['text'][sample_idx]})", fontsize=8)
            plt.ylabel('CTC Posteriors', fontsize=6)
            plt.yticks(list(range(0, 2, 1)))
            plt.tight_layout()
            # if save_path is not None:
            #     plt.savefig(os.path.join(save_path, f'prob.png'))
            reporter.add_figure(f"{tag}/CTC/{sample_id}", figure)
            plt.close()
