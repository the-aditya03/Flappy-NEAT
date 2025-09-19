# Flappy_NEAT
An interactive and AI-powered version of Flappy Bird built with **Pygame** and **NEAT (NeuroEvolution of Augmenting Topologies)**.  

---

## Features

This project lets you:
- **Human Play Mode** – Play the game manually with simple controls  
- **AI Training Mode** – Train birds with NEAT for multiple generations  
- **Training Visualization** – Watch the entire population attempt the game  
- **Best AI Playback** – Replay the top-performing bird after training  

---

## Tech Stack

- Python >= 3.8
- Pygame for game rendering and control
- [NEAT-Python](https://github.com/CodeReclaimers/neat-python) for neuroevolution
- Numpy for vector computations

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/the-aditya03/Flappy-NEAT.git
   cd Flappy_NEAT
   ```

2. Create and activate a virtual environment (recommended):

    for Linux/Mac
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    for Windows
    ```bash
    python3 -m venv venv
    venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. Run the main script
    ```bash
    python3 main.py
    ```

2. Select your desired mode:

    When prompted, enter one of the following modes:
    - `Play` - Control the bird manually via keyboard (space to jump).
    - `Train` - Train the AI using NEAT algorithm without graphical rendering (fastest).
    - `WatchTraining` - Visually watch the NEAT training process (slower).
    - `WatchBest` - Watch the best saved AI genome in action.

---

## Configuration

- **Max Frames per Generation during Training:**  
  The training runs for a maximum of **10,000 frames per generation** (~score of 100). This limits training time per genome and keeps evolution efficient. You can change this value in `main.py` by editing the `MAX_FRAMES_PER_GEN` constant.

- **Number of Generations during Training:**  
  By default, NEAT runs for **50 generations** when training. You can reduce or increase this by modifying the `p.run(...)` call in `main.py` (in eval_genomes function).

- **Rendering during WatchTraining:**  
  For visualization, only **15 generations** are shown by default to keep it manageable. You can change this number in `main.py` where the `eval_genomes` function is invoked with `render=True`.


To customize these parameters, edit the following lines in `main.py`:

```python
MAX_FRAMES_PER_GEN = 10000  # adjust max frames per generation
p.run(eval_genomes, 15)     # change 15 to set number of generations during training
p.run(lambda genomes, config: eval_genomes(genomes, config, render=True), 15)  # change 15 for WatchTraining generations
```

---

## How It Works

- The NEAT algorithm evolves populations of neural networks to maximize a fitness function tied to the bird’s survival and pipe navigation performance.
- The AI's input state for each decision is limited to three normalized values:
  - Vertical distance to the top of the next pipe gap.
  - Vertical distance to the bottom of the next pipe gap.
  - Horizontal distance to the next pipe.
- Bird jump is triggered when the neural network output crosses a threshold.
- Fitness rewards survival time and passing pipes; penalties apply on collisions.

---

## Training Details

- Maximum frames per generation limit training duration for each genome.
- Uses NEAT-Python's `ParallelEvaluator` to evaluate genomes in parallel if available.
- Saves the highest scoring genome after each generation or when a new high score is reached.
- Adjustable parameters in `config-feedforward.txt`.

---

## Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository.
2. Create your branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a Pull Request.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## Acknowledgments

- The NEAT-Python library by CodeReclaimers
- Pygame community for game development resources
- Inspiration from Tim's NEAT Flappy Bird tutorial  
- Open source contributors and the Flappy Bird AI community

---
*✨ Happy evolving and flapping!*  


