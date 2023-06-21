import pandas as pd
from chordgnn.utils.chord_representations import solveChordSegmentation, resolveRomanNumeralCosine, formatRomanNumeral, formatChordLabel, generateRomanText
import copy, os
import numpy as np
import partitura as pt
# Testing with pretrained model
from chordgnn.utils.chord_representations_latest import available_representations
from chordgnn.models.chord import ChordPrediction, PostChordPrediction
import torch
import argparse


parser = argparse.ArgumentParser("Chord Prediction")
parser.add_argument("--use_ckpt", type=str, default="melkisedeath/chord_rec/model-kvd0jic5:v0",
                    help="Wandb artifact to use for prediction")
parser.add_argument("--input_score", type=str, default="./artifacts/op20n3-04.musicxml", help="Path to musicxml input score")

args = parser.parse_args()


artifact_dir = os.path.normpath(f"./artifacts/{os.path.basename(args.use_ckpt)}")
if not os.path.exists(artifact_dir):
    import wandb
    run = wandb.init()
    artifact = run.use_artifact(args.use_ckpt, type='model')
    artifact_dir = artifact.download()

tasks = {
    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35, "tenor": 35,
    "alto": 35, "soprano": 35}
encoder = ChordPrediction(83, 256, tasks, 1, lr=0.0015, dropout=0.44,
                        weight_decay=0.0035, use_nade=False, use_jk=False, use_rotograd=False, device="cpu").module
model = PostChordPrediction(83, 256, tasks, 1, device="cpu", frozen_model=encoder)
model = model.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
encoder = model.frozen_model
model = model.module
score = pt.load_score(args.input_score)
dfdict = {}
with torch.no_grad():
    model.eval()
    encoder.eval()
    prediction = model.predict(encoder.predict(score))

for task in tasks.keys():
    predOnehot = torch.argmax(prediction[task], dim=-1).reshape(-1, 1)
    decoded = available_representations[task].decode(predOnehot)
    dfdict[task] = decoded
dfdict["onset"] = prediction["onset"]
dfdict["s_measure"] = prediction["s_measure"]
df = pd.DataFrame(dfdict)


inputPath = args.input_score
dfout = copy.deepcopy(df)
score = pt.load_score(inputPath)
note_array = score.note_array(include_pitch_spelling=True)
prevkey = ""
bass_part = score.parts[-1]
rn_part = pt.score.Part(id="RNA", part_name="Roman Numerals", quarter_duration=bass_part._quarter_durations[0])
rn_part.add(pt.score.Clef(staff=1, sign="percussion", line=2, octave_change=0), 0)
rn_part.add(pt.score.Staff(number=1, lines=1), 0)

annotations = []
for analysis in dfout.itertuples():
    notes = []
    chord = note_array[(analysis.onset == note_array["onset_beat"]) | (analysis.onset < note_array["onset_beat"]) & (analysis.onset > note_array["onset_beat"] + note_array["duration_beat"])]
    if len(chord) == 0:
        continue
    bass = chord[chord["pitch"] == chord["pitch"].min()]
    thiskey = analysis.localkey
    tonicizedKey = analysis.tonkey
    pcset = analysis.pcset
    numerator = analysis.romanNumeral
    rn2, chordLabel = resolveRomanNumeralCosine(
        analysis.bass,
        analysis.tenor,
        analysis.alto,
        analysis.soprano,
        pcset,
        thiskey,
        numerator,
        tonicizedKey,
    )
    if thiskey != prevkey:
        rn2fig = f"{thiskey}:{rn2}"
        prevkey = thiskey
    else:
        rn2fig = rn2
    formatted_RN = formatRomanNumeral(rn2fig, thiskey)
    annotations.append((formatted_RN, int(bass_part.inv_beat_map(analysis.onset).item())))



annotations = np.array(annotations, dtype=[("rn", "U10"), ("onset_div", "i4")])

# Infer first chord of piece
rn, onset = annotations[0]
annotations["rn"][0] = rn[:rn.index(":")+1] + "V" if rn.lower().endswith("i64") else rn
key = rn[0]
first_notes = np.unique(note_array[note_array["onset_div"] == onset]["step"])
if len(first_notes) > 1:
    pass
else:
    if abs(pt.utils.music.STEPS[first_notes[0].item().capitalize()] - pt.utils.music.STEPS[key.capitalize()])%7 == 3:
        annotations["rn"][0] = rn[:rn.index(":") + 1] + "V"


end_duration = note_array[note_array["onset_div"] == note_array["onset_div"].max()]["duration_div"].max()
bmask = np.array([True] + [(annotations[i]["rn"] != annotations[i-1]["rn"][annotations[i-1]["rn"].index(":")+1:]) if ":" in annotations[i-1]["rn"] else (annotations[i]["rn"] != annotations[i-1]["rn"]) for i in range(1, len(annotations))])
annotations = annotations[bmask]
durations = np.r_[np.diff(annotations["onset_div"]), end_duration]
for i, (rn, onset) in enumerate(annotations):
    note = pt.score.UnpitchedNote(step="F", octave=5, staff=1)
    word = pt.score.Harmony(rn)
    rn_part.add(note, onset, onset+durations[i].item())
    rn_part.add(word, onset)


for item in bass_part.iter_all(pt.score.TimeSignature):
    rn_part.add(item, item.start.t)
for item in bass_part.measures:
    rn_part.add(item, item.start.t, item.end.t)
pt.score.tie_notes(rn_part)

# # TODO: Repair Short Key changes and check correctness.
# rna_annotations = list(rn_part.iter_all(pt.score.Harmony))
# # find indices of rna_annotations text that contain : character
# key_change = np.array([(i, x.text[:x.text.index(':')]) for i, x in enumerate(rna_annotations) if ":" in x.text], dtype=[("idx", "i4"), ("key", "U10")])
# # find where indices are consecutive
# c = np.where(np.diff(key_change["idx"]) < 2)[0] + 1
# c = c[c != key_change["idx"].argmax()]
# problematic_indices = c[np.where(key_change["key"][c+1] == key_change["key"][c-1])]
# key_change_indices = key_change["key"][problematic_indices]
# for idx in key_change_indices:
#     if "/" in rna_annotations[idx].text:
#         rna_annotations[idx].text = rna_annotations[idx].text[rna_annotations[idx].text.index(":")+1:rna_annotations[idx].text.index("/")]
#     else:
#         rna_annotations[idx].text = rna_annotations[idx].text[rna_annotations[idx].text.index(":")+1:] # + "/ degree difference between previous key and current key"

score.parts.append(rn_part)
pt.save_musicxml(score, f"./artifacts/{os.path.splitext(os.path.basename(args.input_score))[0]}_rna.musicxml")