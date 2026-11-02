# demo_data

An example data folder for `labeler.html`: open the tool, choose this folder, and label the six clips.

The clips are the six samples of actor 1001 from **CREMA-D** (Cao et al., "CREMA-D: Crowd-sourced Emotional
Multimodal Actors Dataset", IEEE Transactions on Affective Computing, 2014;
https://github.com/CheyneyComputerScience/CREMA-D), licensed under the Open Database License (ODbL v1.0), renamed to
`clip_0001` … `clip_0006`; two were converted to MP3 to show that MP3 works. The sentence is "Don't forget a jacket"
(CREMA-D's DFA sentence); four clips have it as a transcript, two have none.

`predictions/` holds the real predictions of emotion2vec+ large and MERaLiON-SER-v1 for these clips, taken from
the evaluation in `emotion_recognition/outputs/` (both models call clip_0006, a sad clip, disgusted: predictions are
only guidance).

The labels written here (`annotations/`) are not tracked by git.
