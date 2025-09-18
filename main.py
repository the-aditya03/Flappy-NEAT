import pygame
import game as game_module
import sys
import neat
import numpy as np
import pickle

from collections import deque
from game import Game, Pipe, Background, Base
import config as game_config

BEST_GENOME_PATH = "best_neat_bird.pkl"
MAX_FRAMES_PER_GEN = 10000
NEAT_MIN_INPUTS = 3

pygame.font.init()
FONT = pygame.font.SysFont(None, 40)
SMALL_FONT = pygame.font.SysFont(None, 28)

class SaveBestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self, filename=BEST_GENOME_PATH):
        self.filename = filename
    def post_evaluate(self, config, population, species, best_genome):
        with open(self.filename, "wb") as f:
            pickle.dump(best_genome, f)
        print(f"Saved best genome to {self.filename}")

def get_state_from_pipe(bird, next_pipe):
    vert_top = (next_pipe.height - bird.y) / game_config.SCREEN_HEIGHT
    vert_bottom = ((next_pipe.height + game_config.PIPE_GAP) - bird.y) / game_config.SCREEN_HEIGHT
    horiz = (next_pipe.x - bird.x) / game_config.SCREEN_WIDTH
    return np.array([vert_top, vert_bottom, horiz])

def eval_genome(genome, config):
    bird = game_module.Bird()
    pipes = [Pipe(game_config.SCREEN_WIDTH + 100)]
    bg = Background()
    base = Base()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    frames = 0
    fitness = 0
    score = 0
    running = True

    while running and frames < MAX_FRAMES_PER_GEN:
        frames += 1

        bg.update()
        base.update()
        for pipe in pipes:
            pipe.update()
        pipes = [pipe for pipe in pipes if not pipe.off_screen()]
        if len(pipes) == 0 or pipes[-1].x < game_config.SCREEN_WIDTH - 300:
            pipes.append(Pipe(game_config.SCREEN_WIDTH + 100))

        next_pipe = pipes[0]

        state = get_state_from_pipe(bird, next_pipe)
        output = net.activate(state)
        if output[0] > 0.5:
            bird.jump()
        bird.update()

        fitness += 0.1

        for pipe in pipes:
            if not hasattr(pipe, "passed"):
                pipe.passed = False
            if not pipe.passed and bird.x > pipe.x + game_config.PIPE_WIDTH:
                pipe.passed = True
                score += 1
                fitness += 5

        if check_collision_with_pipes(bird, pipes) or bird.y <= 0 or bird.y >= game_config.SCREEN_HEIGHT - game_config.BASE_HEIGHT:
            fitness -= 1
            running = False

    genome.fitness = fitness
    return fitness

def eval_genomes(genomes, config, render=False):
    if render:
        screen = pygame.display.set_mode((game_config.SCREEN_WIDTH, game_config.SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird AI Training")
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = None

    bg = Background()
    base = Base()
    birds = []
    nets = []
    ge = []

    max_score = -1
    best_genome = None
    best_bird_index = 0

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        bird = game_module.Bird()
        bird.score = 0
        birds.append(bird)
        ge.append(genome)

    pipes = [Pipe(game_config.SCREEN_WIDTH + 100)]
    running = True
    frames = 0

    while running and len(birds) > 0 and frames < MAX_FRAMES_PER_GEN:
        if render:
            clock.tick(game_config.FPS)
        frames += 1

        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        bg.update()
        base.update()
        if render:
            bg.draw(screen)

        if len(pipes) == 0 or pipes[-1].x < game_config.SCREEN_WIDTH - 300:
            pipes.append(Pipe(game_config.SCREEN_WIDTH + 100))

        next_pipe = pipes[0]
        remove_indices = []
        best_score = -1

        for i, bird in enumerate(birds):
            state = get_state_from_pipe(bird, next_pipe)
            output = nets[i].activate(state)
            if output[0] > 0.5:
                bird.jump()
            bird.update()
            if render:
                bird.draw(screen)
            ge[i].fitness += 0.1

            if check_collision_with_pipes(bird, pipes) or bird.y <= 0 or bird.y >= game_config.SCREEN_HEIGHT - game_config.BASE_HEIGHT:
                ge[i].fitness -= 1
                remove_indices.append(i)
            if bird.score > best_score:
                best_score = bird.score
                best_bird_index = i

        if best_score > max_score:
            max_score = best_score
            best_genome = ge[best_bird_index]

        if render and birds:
            best_bird = birds[best_bird_index]
            pygame.draw.circle(screen, (255, 0, 0), (best_bird.x + game_config.BIRD_SIZE // 2, int(best_bird.y + game_config.BIRD_SIZE // 2)), 20, 3)

        for index in sorted(remove_indices, reverse=True):
            birds.pop(index)
            nets.pop(index)
            ge.pop(index)

        for pipe in pipes:
            pipe.update()
            if render:
                pipe.draw(screen)
            for i, bird in enumerate(birds):
                if not hasattr(pipe, "passed"):
                    pipe.passed = False
                if not pipe.passed and bird.x > pipe.x + game_config.PIPE_WIDTH:
                    pipe.passed = True
                    bird.score += 1
                    ge[i].fitness += 5

        pipes = [pipe for pipe in pipes if not pipe.off_screen()]
        if render:
            base.draw(screen)
            alive_text = FONT.render(f"Birds alive: {len(birds)}", True, (255, 255, 255))
            max_score_text = FONT.render(f"Max score: {max_score}", True, (255, 255, 255))
            screen.blit(alive_text, (10, 10))
            screen.blit(max_score_text, (10, 50))
            pygame.display.flip()

    if frames >= MAX_FRAMES_PER_GEN:
        print(f"[Timeout] Generation ended after {MAX_FRAMES_PER_GEN} frames")
        if birds:
            best_genome = ge[best_bird_index]
            with open(BEST_GENOME_PATH, "wb") as f:
                pickle.dump(best_genome, f)
            print("[Timeout Save] Best genome saved at timeout")

def run_neat(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(SaveBestGenomeReporter(BEST_GENOME_PATH))

    try:
        from neat.parallel import ParallelEvaluator
        pe = ParallelEvaluator(4, eval_genome)
        winner = p.run(pe.evaluate, 50)
    except Exception:
        print("Parallel evaluation unavailable, using serial evaluation.")
        winner = p.run(lambda g, c: eval_genomes(g, c, render=False), 50)
    return winner

def playback_neat_best(screen, clock, font):
    try:
        with open(BEST_GENOME_PATH, 'rb') as f:
            winner = pickle.load(f)
    except FileNotFoundError:
        print("No saved NEAT best genome found. Please train first.")
        return

    config_path = "config-feedforward.txt"
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    game_instance = game_module.Game()
    running = True

    while running:
        clock.tick(game_config.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game_instance.check_collision():
            running = False
            break

        next_pipe = game_instance.get_next_pipe(game_instance.bird)
        state = get_state_from_pipe(game_instance.bird, next_pipe)
        output = net.activate(state)
        if output[0] > 0.5:
            game_instance.bird.jump()

        game_instance.update()
        game_instance.draw(screen, font)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

def check_collision_with_pipes(bird, pipes):
    bird_rect = bird.get_rect()
    if bird.y <= 0 or bird.y + game_config.BIRD_SIZE >= game_config.SCREEN_HEIGHT - game_config.BASE_HEIGHT:
        return True
    for pipe in pipes:
        top, bottom = pipe.get_rects()
        if bird_rect.colliderect(top) or bird_rect.colliderect(bottom):
            return True
    return False

def show_start_screen(screen, clock, font, small_font, max_score):
    bg = Background()  
    bg.draw(screen)

    overlay = pygame.Surface((game_config.SCREEN_WIDTH, game_config.SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))

    screen.blit(overlay, (0, 0))

    title_text = font.render("Flappy Bird NEAT", True, (255, 255, 255))
    instr_text = small_font.render("Press SPACE to start playing", True, (255, 255, 255))
    best_score_text = small_font.render(f"Your best score: {max_score}", True, (255, 255, 0))

    screen.blit(title_text, ((game_config.SCREEN_WIDTH - title_text.get_width()) // 2, game_config.SCREEN_HEIGHT // 3))
    screen.blit(instr_text, ((game_config.SCREEN_WIDTH - instr_text.get_width()) // 2, game_config.SCREEN_HEIGHT // 2))
    screen.blit(best_score_text, ((game_config.SCREEN_WIDTH - best_score_text.get_width()) // 2, game_config.SCREEN_HEIGHT // 2 + 40))

    pygame.display.flip()

    waiting = True
    while waiting:
        clock.tick(game_config.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False

def show_game_over(game, screen, clock, font, small_font, max_score):
    overlay = pygame.Surface((game_config.SCREEN_WIDTH, game_config.SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    go_text = font.render("Game Over", True, (255, 50, 50))
    score_text = font.render(f"Your Score: {game.score}", True, (255, 255, 255))
    max_score_text = small_font.render(f"Best Score: {max_score}", True, (255, 215, 0))

    instr_text = small_font.render("Press SPACE to restart or ESC to quit", True, (255, 255, 255))

    screen.blit(go_text, ((game_config.SCREEN_WIDTH - go_text.get_width()) // 2, (game_config.SCREEN_HEIGHT // 2) - 80))
    screen.blit(score_text, ((game_config.SCREEN_WIDTH - score_text.get_width()) // 2, (game_config.SCREEN_HEIGHT // 2) - 30))
    screen.blit(max_score_text, ((game_config.SCREEN_WIDTH - max_score_text.get_width()) // 2, (game_config.SCREEN_HEIGHT // 2) + 10))
    screen.blit(instr_text, ((game_config.SCREEN_WIDTH - instr_text.get_width()) // 2, (game_config.SCREEN_HEIGHT // 2) + 60))

    pygame.display.flip()

    waiting = True
    while waiting:
        clock.tick(game_config.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

def run_human_game(game_instance, screen, clock, font, small_font):
    max_score = 0
    show_start_screen(screen, clock, font, small_font, max_score)

    running = True
    while running:
        clock.tick(game_config.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game_instance.bird.jump()
                if event.key == pygame.K_ESCAPE:
                    running = False

        game_instance.update()
        if game_instance.check_collision():
            if game_instance.score > max_score:
                max_score = game_instance.score
            show_game_over(game_instance, screen, clock, font, small_font, max_score)
            game_instance.reset()
            show_start_screen(screen, clock, font, small_font, max_score)
            continue

        game_instance.draw(screen, font)
        pygame.display.flip()

        if game_instance.score > 1 and game_instance.score % 10 == 0:
            game_config.FPS += 0.5

def main(mode="Play"):
    pygame.init()
    screen = pygame.display.set_mode((game_config.SCREEN_WIDTH, game_config.SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")
    game_module.load_assets()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 40)
    small_font = pygame.font.SysFont(None, 28)

    if mode == "Play":
        game_instance = game_module.Game()
        run_human_game(game_instance, screen, clock, font, small_font)
    elif mode == "Train":
        run_neat("config-feedforward.txt")
    elif mode == "WatchTraining":
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "config-feedforward.txt"
        )
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())
        p.run(lambda genomes, config: eval_genomes(genomes, config, render=True), 15)
    elif mode == "WatchBest":
        playback_neat_best(screen, clock, font)
    else:
        print("Invalid mode! Choose 'Me', 'AI', or 'Watch'.")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mode = input("What are you looking for? (Play, Train, WatchTraining, WatchBest): ").strip()
    main(mode)
