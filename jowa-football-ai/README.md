# Jowa Football AI v2

**6v6 DQN-Powered Football Game — Open Source**  
By **AXV (Ahmed Walid)** · 2026

---

## What is Jowa Football AI v2?

Jowa Football AI v2 is an open-source 6v6 football simulation game featuring intelligent AI players powered by **Deep Q-Network (DQN)** reinforcement learning using PyTorch. Two teams (Red vs Blue) play in a Formation 3-2-1. You can take control of any player at any time while the AI handles the rest.

---

## What's New in v2

- Upgraded from 3v3 to **6v6** with Formation 3-2-1
- AI upgraded from Q-table to **DQN (Deep Q-Network)** using PyTorch
- Shared replay buffer with capacity 200,000 transitions
- Player selection system — pick any player with ↑↓ then Enter
- Auto epsilon decay with periodic model saving every 3600 frames
- CUDA support — runs on GPU if available

---

## Game Info

| Property | Value |
|----------|-------|
| Mode | 6v6 (Red vs Blue) |
| Formation | 3-2-1 |
| AI Type | DQN — Deep Q-Network (PyTorch) |
| Actions | 9 — Stop, N, S, E, W, NE, NW, SE, SW |
| Resolution | 800 × 600 |
| FPS | 60 |
| Device | CUDA (GPU) or CPU — auto-detected |
| Model Files | football_model_red.pth, football_model_blue.pth |

---

## Requirements

- Python 3.8+
- pygame
- PyTorch
- NumPy

Install all dependencies:

```bash
pip install pygame torch numpy
```

---

## How to Run

1. Download `jowa-football-ai.zip`
2. Extract the ZIP file
3. Open a terminal in the extracted folder
4. Run:

```bash
python jowa-football-ai-with-players_PLAYMODEAI4_6p.py
```

---

## Controls

### Player Selection
| Key | Action |
|-----|--------|
| ↑ / ↓ | Cycle through players |
| Enter / Space | Control selected player |
| ESC | Back to selection mode |

### Movement (after selecting a player)
| Key | Direction |
|-----|-----------|
| W or ↑ | Up |
| S or ↓ | Down |
| A or ← | Left |
| D or → | Right |

---

## How the AI Works

The AI uses **DQN (Deep Q-Network)**:

1. Each player's state is encoded — ball position, player positions, velocities
2. The DQN maps `state → Q-values` for all 9 possible actions
3. The agent picks the action with the highest Q-value (or explores randomly based on epsilon)
4. After each step, a `(state, action, reward, next_state, done)` tuple is pushed to a shared replay buffer
5. The network trains on random batches sampled from the buffer using the Bellman equation
6. A target network updates softly via tau (`τ = 0.005`) for stable training
7. Epsilon decays over time — AI becomes less random and more strategic
8. Model weights are auto-saved every 3600 frames to `football_model_red.pth` / `football_model_blue.pth`

### DQN Architecture

```
Input (state_size)
→ Linear(256) + ReLU
→ Linear(256) + ReLU
→ Linear(128) + ReLU
→ Linear(9)  ← Q-values for each action
```

---

## Project Structure

```
jowa-football-ai/
├── jowa-football-ai-with-players_PLAYMODEAI4_6p.py   # Main game file
├── football_model_red.pth                             # Red team model (auto-generated)
├── football_model_blue.pth                            # Blue team model (auto-generated)
└── README.md                                          # This file
```

---

## License

This project is **open source** — free to use, modify, and distribute.  
© 2025/2026 AXV (Ahmed Walid)

---

## Links

- Website: [IFelx Web](https://ahmedxv-v12.github.io)
- Instagram: [@axv.bin](https://www.instagram.com/axv.bin)