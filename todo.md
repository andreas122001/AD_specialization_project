# TODO:

## PAPER

HOW CAN WE FIX THE RESULTS?
- Need more runs (3 trainings * 3 evals = 9 runs per experiment)
- Which experiments to focus on?
    - Do we do more sequence scaling tests?
    - Do we test parameter scaling?
    - How can we ablate gating mechanism? (And what to do if it doesn't work?)
    - Focus only on a set seq + param and do an ensamble?




## OLD

[x] Bugfix
- [x] Use correct temporal stride for evaluation (has a higher fps)
    - [x] Did fixing this impact performance?
    - Yes, kinda, a bit, I think?
- [x] Change colors of the attention visualization

Static memory consumption (2 A100m40 gpus, LB5): 28-36 Gi / GPU


# V1:
[] Train:
- [x] default: original tfpp, reproduced
- [x] static-LB2s1-L2: static with 2 samples, 250ms lookback
- [x] static-LB5s1-L2: static with 5 samples, 1s lookback
- [x] static-LB5s2-L2: static with 5 samples, 2s lookback
- [x] LB2s1-L2: recurrent with 2 samples, 250ms lookback
- [ ] LB5s1-L2: recurrent with 3 samples, 500ms lookback
- [x] static-LB5s1-L8: larger fusor (8 layers)
- [x] static-LB5s1-L2-noself: no self-attn
- [-] static-LB5s1-L2-notraj: w/o trajectory decoder
- [-] static-LB5s1-L2-nomask: w/o ego velocity mask
- [x] static-LB5s1-L2-h1: single-head fusion layer
- [1] lidar-LB2s1
- [1] lidar-LB5s1
[ ] Bench2Drive:
- [-] Default
- [-] static-LB2s1-L2
- [ ] static-LB5s1-L2
- [ ] LB2s1-L2
- [ ] Static: LB2, LB5
- [ ] Recurrent: LB1
[ ] Short routes:
- [x] Default
- [x] static-LB2s1-L2
- [ ] static-LB5s1-L2
- [ ] LB2s1-L2
- [ ] Static: LB2, LB5
- [ ] Recurrent: LB1

# V2:
- [x] static-LB5s1
- [x] static-LB9s1*
- [x] static-LB9s2*
- [x] LB5s2
- [x] lidar-LB5s1
- [x] static-LB5s1-notraj
- [x] static-LB5s1-notraj
- [x] static-LB5s1-noprune

*=bad confidences

[x] Find out what is wrong with trajectory prediction
- [x] Implement multi-modality
- [x] compare with/without speed token
- [x] compare different decoder sizes

[ ] Analyse performance of temporal module
- [x] Make plot of scenarios against each other (default vs temporal)
- [ ] Benchmark on Bench2Drive

[ ] Writing:
- [ ] Intro, theory
- [ ] Literature
- [ ] Methodology

[ ] Find what else to do?
- [ ] Adding other scenarios (or taking from ReasonNet?)
- [ ] Running in "realistic" simulator?
- [x] Attention visualization

[ ] Scientific contribution:
- [ ] How can temporal fusion improve performance?
- [ ] Can the agent use the temporal information (is it paying attention)?
- [ ] Can we use static fusion (semi-recurrent) over full recurrence?
- [ ] How important is trajectory prediction for forcing temporal info usage?
- [ ] Are Multi-task learning (MTL) models (like TF++) still relevant?

Ablations (what to do?):
- Masking speed token vs not?
- Temporal module size (n_layers, say 2 vs 8)

# Static-LB5s1-L2: 
Overtaking :  0.5555555555555556
Merging :  0.4875
Emergency_Brake :  0.8333333333333334
Give_Way :  0.6
Traffic_Signs :  0.7947368421052632
Mean:0.6542251461988304
