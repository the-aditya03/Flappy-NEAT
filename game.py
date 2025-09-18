import pygame
import config
import random
from collections import deque

_bg_image = None
_base_image = None
_pipe_image_full = None
_bird_image_scaled = None

def load_assets():
    global _bg_image, _base_image, _pipe_image_full, _bird_image_scaled
    _bg_image = pygame.image.load(config.BG_IMG).convert()
    _bg_image = pygame.transform.smoothscale(_bg_image, (config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    _base_image = pygame.image.load(config.BASE_IMG).convert_alpha()
    _base_image = pygame.transform.smoothscale(_base_image, (config.SCREEN_WIDTH, config.BASE_HEIGHT))

    _pipe_image_full = pygame.image.load(config.PIPE_IMG).convert_alpha()

    bird_image = pygame.image.load(config.BIRD_IMG).convert_alpha()
    _bird_image_scaled = pygame.transform.smoothscale(bird_image, (config.BIRD_SIZE, config.BIRD_SIZE))

class Background:
    def __init__(self):
        self.image = _bg_image
        self.x = 0

    def update(self):
        self.x -= 1
        if self.x <= -config.SCREEN_WIDTH:
            self.x = 0

    def draw(self, screen):
        screen.blit(self.image, (self.x, 0))
        screen.blit(self.image, (self.x + config.SCREEN_WIDTH, 0))

class Base:
    def __init__(self):
        self.image = _base_image
        self.x = 0
        self.y = config.SCREEN_HEIGHT - config.BASE_HEIGHT + 20
        self.speed = config.PIPE_SPEED

    def update(self):
        self.x -= self.speed
        if self.x <= -config.SCREEN_WIDTH:
            self.x = 0

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))
        screen.blit(self.image, (self.x + config.SCREEN_WIDTH, self.y))

class Bird:
    def __init__(self):
        self.x = config.BIRD_X
        self.y = config.SCREEN_HEIGHT // 2
        self.velocity = 0
        self.image = _bird_image_scaled
        self.score = 0 
        self.passed_pipes = 0

    def update(self):
        self.velocity += config.GRAVITY
        self.y += self.velocity

    def jump(self):
        self.velocity = -config.JUMP_STRENGTH

    def draw(self, screen):
        screen.blit(self.image, (self.x, int(self.y)))

    def get_rect(self):
        return pygame.Rect(self.x, int(self.y), config.BIRD_SIZE, config.BIRD_SIZE)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(100, config.SCREEN_HEIGHT - config.PIPE_GAP - 100)
        self.passed = False

        self.bottom_image_full = _pipe_image_full
        self.top_image_full = pygame.transform.flip(self.bottom_image_full, False, True)
        self.top_image = pygame.transform.smoothscale(self.top_image_full, (config.PIPE_WIDTH, self.height))
        bottom_height = config.SCREEN_HEIGHT - self.height - config.PIPE_GAP
        self.bottom_image = pygame.transform.smoothscale(self.bottom_image_full, (config.PIPE_WIDTH, bottom_height))

    def update(self):
        self.x -= config.PIPE_SPEED

    def draw(self, screen):
        screen.blit(self.top_image, (self.x, 0))
        screen.blit(self.bottom_image, (self.x, self.height + config.PIPE_GAP))

    def get_rects(self):
        top = pygame.Rect(int(self.x), 0, config.PIPE_WIDTH, self.height)
        bottom = pygame.Rect(int(self.x), self.height + config.PIPE_GAP, config.PIPE_WIDTH, config.SCREEN_HEIGHT - self.height - config.PIPE_GAP)
        return top, bottom
    
    def off_screen(self):
        return self.x + config.PIPE_WIDTH < 0

class Game:
    def __init__(self):
        self.bg = Background()
        self.base = Base()
        self.bird = Bird()
        self.pipes = deque()
        self.score = 0
        self.spawn_pipe()
        self.spawn_pipe(offset=200)

    def spawn_pipe(self, offset=0):
        self.pipes.append(Pipe(config.SCREEN_WIDTH + offset))

    def update(self, bird=None, action=None):
        if bird is None:
            bird = self.bird

        self.bg.update()
        self.base.update()

        if action == 1:
            bird.jump()

        bird.update()

        for pipe in self.pipes:
            pipe.update()

        while self.pipes and self.pipes[0].x + config.PIPE_WIDTH < 0:
            self.pipes.popleft()

        while len(self.pipes) < 2:
            self.spawn_pipe()

        for pipe in self.pipes:
            if not pipe.passed and pipe.x + (config.PIPE_WIDTH / 2) < bird.x:
                pipe.passed = True
                bird.score += 1
                if bird == self.bird:
                    self.score += 1

    def check_collision(self, bird=None):
        if bird is None:
            bird = self.bird

        bird_rect = bird.get_rect()
        if bird.y <= 0 or bird.y + config.BIRD_SIZE >= self.base.y:
            return True

        for pipe in self.pipes:
            top, bottom = pipe.get_rects()
            if bird_rect.colliderect(top) or bird_rect.colliderect(bottom):
                return True
        return False

    def get_next_pipe(self, bird):
        for pipe in self.pipes:
            if pipe.x + config.PIPE_WIDTH > bird.x:
                return pipe
        return self.pipes[0]

    def draw(self, screen, font, birds=None):
        self.bg.draw(screen)
        for pipe in self.pipes:
            pipe.draw(screen)

        if birds:
            for bird in birds:
                bird.draw(screen)
                bird_score_text = font.render(str(int(bird.score)), True, (0, 0, 0))
                screen.blit(bird_score_text, (bird.x, int(bird.y) - 20))
        else:
            self.bird.draw(screen)

        self.base.draw(screen)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

    def reset(self):
        self.bird = Bird()
        self.pipes = deque()
        self.score = 0
        config.FPS = 60
        config.PIPE_GAP = 150
        self.spawn_pipe()
        self.spawn_pipe(offset=200)
