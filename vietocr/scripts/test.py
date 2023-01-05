from os import path
from tqdm import tqdm
from torch.utils.data import DataLoader

from .. import const
from ..tool.translate import build_model
from ..tool import utils, stats
from ..loader.dataloader import (
    OCRDataset,
    Collator,
    ClusterRandomSampler
)


def main(config, options):
    batch_size = options.batch_size
    annotation_path = options.test_annotation

    model, vocab = build_model(config)
    model.eval()

    lmdb_path = path.join(
        const.lmdb_dir,
        utils.annotation_uuid(annotation_path)
    )

    data_root = path.dirname(annotation_path)
    annotation_path = path.basename(annotation_path)

    # TODO: rewrite dataset so that the image can be loaded without
    # using a dataloader, collator and sampler
    dataset = OCRDataset(lmdb_path=lmdb_path,
                         root_dir=data_root,
                         annotation_path=annotation_path,
                         vocab=vocab,
                         image_height=config['image_height'],
                         image_min_width=config['image_min_width'],
                         image_max_width=config['image_max_width'])

    sampler = ClusterRandomSampler(
        dataset,
        batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=Collator(),
        shuffle=False,
        drop_last=False,
    )

    # Perform testing
    acc_per_char = stats.AverageStatistic()
    acc_full_seq = stats.AverageStatistic()
    all_pr_sents = []
    all_gt_sents = []
    device = next(model.parameters()).device
    for batch in tqdm(test_loader):
        img = batch['img'].to(device)
        tgt_output = batch['tgt_output'].to(device)

        # Predict, no teacher forcing
        # the tgt output is only for seq length
        outputs = model(img, tgt_output)
        probs, translated = outputs.topk(k=1, dim=-1)
        # TODO: add confidence to prediction
        # perferably to the predictor, so the the outputs
        # are united, and there's no mismatch between codes
        # probs = probs.squeeze(-1)
        translated = translated.squeeze(-1)

        # Validation accuracy
        pr_sents = vocab.batch_decode(translated.tolist())
        gt_sents = vocab.batch_decode(tgt_output.tolist())
        acc_pc = utils.compute_accuracy(pr_sents, gt_sents, 'per_char')
        acc_fs = utils.compute_accuracy(pr_sents, gt_sents, 'full_sequence')

        # save results
        # all_pr_sents.extend(pr_sents)
        # all_gt_sents.extend(gt_sents)
        tbp = [
            f"GT:   {GT}\nPR:   {PR}"
            for GT, PR in zip(gt_sents, pr_sents)
        ]
        tqdm.write(("\n~~~~~~~~~\n").join(tbp))

        # Statistic
        acc_per_char.append(acc_pc)
        acc_full_seq.append(acc_fs)

    tqdm.write(f"Accuracy per character: {acc_per_char.summarize()}")
    tqdm.write(f"Accuracy full sequence: {acc_full_seq.summarize()}")
