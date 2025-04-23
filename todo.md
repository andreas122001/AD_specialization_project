# TODO:

[x] Bugfix
- [x] Use correct temporal stride for evaluation (has a higher fps)
    - [x] Did fixing this impact performance?
    - Yes, kinda, a bit, I think?
- [x] Change colors of the attention visualization

[ ] Find out what is wrong with trajectory prediction
- [ ] Implement multi-modality
- [ ] compare with/without speed token
- [ ] compare different decoder sizes

[ ] Analyse performance of temporal module
- [ ] Make plot of scenarios against each other (default vs temporal)
- [ ] Benchmark on Bench2Drive

[ ] Writing:
- [ ] Intro, theory
- [ ] Literature
- [ ] Methodology

[ ] Find what else to do?
- [ ] Adding other scenarios (or taking from ReasonNet?)
- [ ] Running in "realistic" simulator?
- [ ] Attention visualization

Ablations (what to do?):
- Masking speed token vs not?
- Temporal module size (n_layers, say 2 vs 8)
