import chordgnn as st
import torch
import random
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument("--collection", type=str, default="all",
                choices=["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"],  help="Collection to test on.")
parser.add_argument("--predict", action="store_true", help="Obtain Predictions using wandb cloud stored artifact.")
parser.add_argument('--use_nade', action="store_true", help="Use NADE instead of MLP classifier.")
parser.add_argument('--use_jk', action="store_true", help="Use Jumping Knowledge In graph Encoder.")
parser.add_argument('--use_rotograd', action="store_true", help="Use Rotograd for MTL training.")
parser.add_argument("--include_synth", action="store_true", help="Include synthetic data.")
parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
parser.add_argument("--use_ckpt", type=str, default=None, help="Use checkpoint for prediction.")
parser.add_argument("--task", type=str, default="localkey", help="Which task to train on")

# for reproducibility
# torch.manual_seed(0)
# random.seed(0)
# torch.use_deterministic_algorithms(True)


args = parser.parse_args()
if isinstance(eval(args.gpus), int):
    if eval(args.gpus) >= 0:
        devices = [eval(args.gpus)]
        dev = devices[0]
    else:
        devices = None
        dev = "cpu"
else:
    devices = [eval(gpu) for gpu in args.gpus.split(",")]
    dev = None
n_layers = args.n_layers
n_hidden = args.n_hidden
force_reload = False
num_workers = args.num_workers


name = "{}-{}x{}-lr={}-wd={}-dr={}".format(f"SingleTask-{args.task}", n_layers, n_hidden,
                                            args.lr, args.weight_decay, args.dropout)

wandb_logger = WandbLogger(
    log_model=True,
    entity="melkisedeath",
    project="chord_single_task",
    group=args.collection,
    job_type=args.task,
    name=name)


# datamodule = GraphMixVSDataModule(
#     batch_size=1, num_workers=num_workers,
#     force_reload=force_reload, test_collections=collections,
#     pot_edges_max_dist=pot_edges_max_dist)
# datamodule.setup()


datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=16, include_synth=args.include_synth, num_tasks=args.task, collection=args.collection, batch_size=args.batch_size)
model = st.models.chord.SingleTaskPrediction(
    datamodule.features, args.n_hidden, datamodule.tasks, args.n_layers, lr=args.lr, dropout=args.dropout,
    weight_decay=args.weight_decay, use_nade=args.use_nade, use_jk=args.use_jk, use_rotograd=args.use_rotograd,
    device=dev
    )

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.02, patience=5, verbose=False, mode="min")
use_ddp = len(devices) > 1 if isinstance(devices, list) else False
trainer = Trainer(
    max_epochs=50,
    accelerator="auto", devices=devices,
    num_sanity_val_steps=1,
    logger=wandb_logger,
    plugins=DDPPlugin(find_unused_parameters=False) if use_ddp else None,
    # callbacks=[early_stop_callback],
    )

if not args.predict:
    if args.use_ckpt is not None:
        import wandb, os
        run = wandb.init(
            project="chord_rec",
            entity="melkisedeath",
            job_type="hgraph",
            group="Full dataset",
            name=name)
        artifact = run.use_artifact(f'melkisedeath/chord_rec/model-{args.use_ckpt}:v0', type='model')
        artifact_dir = artifact.download()
        print("Using artifact from: ", artifact_dir)
        trainer.test(model, datamodule, ckpt_path=os.path.join(artifact_dir, "model.ckpt"))
    else:
        # training
        trainer.fit(model, datamodule)
        # Testing with best model
        trainer.test(model, datamodule)