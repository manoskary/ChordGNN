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
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.44)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0015)
parser.add_argument('--weight_decay', type=float, default=0.0035)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument("--collection", type=str, default="all",
                choices=["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"],  help="Collection to test on.")
parser.add_argument("--predict", action="store_true", help="Obtain Predictions using wandb cloud stored artifact.")
parser.add_argument('--use_jk', action="store_true", help="Use Jumping Knowledge In graph Encoder.")
parser.add_argument('--mtl_norm', default="none", choices=["none", "Rotograd", "NADE", "GradNorm", "Neutral"], help="Which MLT optimization to use.")
parser.add_argument("--include_synth", action="store_true", help="Include synthetic data.")
parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
parser.add_argument("--use_ckpt", type=str, default=None, help="Use checkpoint for prediction.")
parser.add_argument("--num_tasks", type=int, default=11, choices=[5, 11, 14], help="Number of tasks to train on.")
parser.add_argument("--data_version", type=str, default="v1.0.0", choices=["v1.0.0", "latest"], help="Version of the dataset to use.")
parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs to train for.")

# for reproducibility
torch.manual_seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)


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

first_name = args.mtl_norm if args.mtl_norm != "none" else "Wloss"
name = "{}-{}x{}-lr={}-wd={}-dr={}".format(first_name, n_layers, n_hidden,
                                            args.lr, args.weight_decay, args.dropout)
use_nade = args.mtl_norm == "NADE"
use_rotograd = args.mtl_norm == "Rotograd"
use_gradnorm = args.mtl_norm == "GradNorm"
weight_loss = args.mtl_norm not in ["Neutral", "Rotograd", "GradNorm"]

wandb_logger = WandbLogger(
    log_model=True,
    entity="melkisedeath",
    project="chord_rec",
    group=args.collection,
    # group="ablation",
    job_type=f"data={args.data_version}",
    # job_type=first_name,
    name=name)

datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=16, include_synth=args.include_synth, num_tasks=args.num_tasks,
    collection=args.collection, batch_size=args.batch_size, version=args.data_version)
model = st.models.chord.ChordPrediction(
    datamodule.features, args.n_hidden, datamodule.tasks, args.n_layers, lr=args.lr, dropout=args.dropout,
    weight_decay=args.weight_decay, use_nade=use_nade, use_jk=args.use_jk, use_rotograd=use_rotograd,
    use_gradnorm=use_gradnorm, device=dev, weight_loss=weight_loss)
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="global_step", mode="max")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.02, patience=5, verbose=False, mode="min")
use_ddp = len(devices) > 1 if isinstance(devices, list) else False
trainer = Trainer(
    max_epochs=args.n_epochs,
    accelerator="auto", devices=devices,
    num_sanity_val_steps=1,
    logger=wandb_logger,
    plugins=DDPPlugin(find_unused_parameters=False) if use_ddp else None,
    callbacks=[checkpoint_callback],
    reload_dataloaders_every_n_epochs=5,
    )

if not args.predict:
    if args.use_ckpt is not None:
        import wandb, os
        run = wandb.init(
            project="chord_rec",
            entity="melkisedeath",
            group=args.collection,
            job_type=f"data={args.data_version}",
            name=name)
        artifact = run.use_artifact(args.use_ckpt, type='model')
        artifact_dir = artifact.download()
        print("Using artifact from: ", artifact_dir)
        trainer.test(model, datamodule, ckpt_path=os.path.join(artifact_dir, "model.ckpt"))
    else:
        # training
        trainer.fit(model, datamodule)
        # Testing with best model
        trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)
else:
    # Testing with pretrained model
    import wandb
    import pandas as pd
    # from chordgnn.utils.chord_representations import available_representations, solveChordSegmentation, resolveRomanNumeralCosine, formatRomanNumeral, formatChordLabel, generateRomanText
    from chordgnn.utils.chord_representations_latest import available_representations
    import partitura as pt
    import copy, os
    import numpy as np
    import music21

    artifact_dir = os.path.normpath("./artifacts/model-i7sxzy4y:v0")
    if not os.path.exists(artifact_dir):
        run = wandb.init()
        artifact = run.use_artifact(args.use_ckpt, type='model')
        artifact_dir = artifact.download()
    model = model.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt")).module
    inputPath = "./artifacts/op20n3-04.krn"
    filename = os.path.basename(inputPath).split(".")[0] + "-chords.csv"
    score = pt.load_score(inputPath)
    dfdict = {}
    with torch.no_grad():
        model.eval()
        prediction = model.predict(score)

    for task in datamodule.tasks.keys():
        predOnehot = torch.argmax(prediction[task], dim=-1).reshape(-1, 1)
        decoded = available_representations[task].decode(predOnehot)
        dfdict[task] = decoded
    dfdict["onset"] = prediction["onset"]
    dfdict["s_measure"] = prediction["s_measure"]
    df = pd.DataFrame(dfdict)
    # Save the predictions to new artifacts folder
    if not os.path.exists("./artifacts"):
        os.mkdir("./artifacts")
    df.to_csv(os.path.join("./artifacts", filename), index=False)




