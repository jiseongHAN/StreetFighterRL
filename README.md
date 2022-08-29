# StreetFigther II' (Beta)

This is a private project to make StreetFighter Agent.

It consists of training an agent to clear StreetFighter with deep reinforcement learning methods.

This agent based on [Super-mario-rl](https://github.com/jiseongHAN/Super-Mario-RL.git)

<p float="center">
  <img src="asset/ex1.gif" width="350" />
  <img src="asset/ex2.gif" width="350" /> 
</p>

# Get started


## Cloning git

```
git clone https://github.com/jiseongHAN/StreetFighterRL.git
cd StreetFighterRL
```


## Install Requirements
```
pip install -r requirements.txt
```

## Install StreetFighter

download rom file from https://edgeemu.net/details-12765.htm

and import streetfighter rom file on gym retro
```
python3 -m retro.import <path_to_your_ROMs_directory>
```

# Running

## Train

* Train with dueling dqn.
```
python duel_dqn.py
```

### Result
* *.pth : save weight of q, q_target every 50 training


## Evaluate
* Test and render trained agent.
* To test our agent, we need 'q_target.pth' that generated at the training step.
```
python eval.py
```
* Or you can use your own agent.
```
python eval.py your_own_agent.pth
```


## TODO: Add prev action for train to use combo

## Reference
[Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.](https://arxiv.org/pdf/1511.06581.pdf)

[Ryuforcement](https://github.com/Camille-Gouneau/Ryuforcement)
