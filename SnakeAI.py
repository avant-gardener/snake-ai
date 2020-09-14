""" Snake game """
import random
import time
import math
import pygame
import numpy as np
import NeuralNetwork
import SnakeTrainer
import Settings

red = (200, 30, 30)
black = (0, 0, 0)
gray = (70, 70, 70)
window_color = (200, 200, 200)
snake_speed = Settings.snake_speed
display_width = Settings.display_width
display_height = Settings.display_height
block_size = Settings.block_size
block_count = Settings.block_count
display = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()

pygame.init()


def draw_grid():
    for x in range(display_width):
        for y in range(display_height):
            rect = pygame.Rect(x * block_size, y * block_size,
                               block_size, block_size)
            pygame.draw.rect(display, gray, rect, 1)


def display_snake(snake_pos):
    color = black
    snake_size_x = block_size - 4
    snake_size_y = block_size - 4
    for position in snake_pos:
        if position == snake_pos[0]:
            pygame.draw.rect(display, color,
                             pygame.Rect(position[0], position[1],
                                         block_size, block_size))
        else:
            pygame.draw.rect(display, color,
                             pygame.Rect(position[0] + 2, position[1] + 2,
                                         snake_size_x, snake_size_y))


def display_apple(apple_pos):
    pygame.draw.rect(display, red,
                     pygame.Rect(apple_pos[0] + 2, apple_pos[1] + 2,
                                 block_size - 4, block_size - 4))


def update_snake(snake_head_pos, snake_pos, apple_pos, direction, cur_direction):
    if cur_direction != direction:
        if is_on_grid(snake_head_pos):
            cur_direction = direction
    if cur_direction == "right":
        snake_head_pos[0] += snake_speed
    elif cur_direction == "left":
        snake_head_pos[0] -= snake_speed
    elif cur_direction == "up":
        snake_head_pos[1] -= snake_speed
    elif cur_direction == "down":
        snake_head_pos[1] += snake_speed
    snake_pos.insert(0, list(snake_head_pos))
    snake_pos.pop()

    return snake_pos, apple_pos, cur_direction


def collision_with_boundaries(snake_h):
    if snake_h[0] >= display_height or snake_h[0] < 0 or \
       snake_h[1] >= display_width or snake_h[1] < 0:
        return 1
    return 0


def collision_with_self(snake_pos):
    snake_h = snake_pos[0]
    snake_crashed = 0
    for snake_part in snake_pos[24:]:
        if snake_part[0] <= snake_h[0] < snake_part[0] + block_size and \
           snake_part[1] <= snake_h[1] < snake_part[1] + block_size:
            snake_crashed = 1
        elif snake_part == snake_pos[-1]:
            if snake_h[0] <= snake_part[0] < snake_h[0] + block_size and \
               snake_h[1] < snake_part[1] < snake_h[1] + block_size:
                snake_crashed = 1
    return snake_crashed


def collision_with_apple(t_score, snake_pos):
    apple_pos = [random.randrange(block_count) * block_size, random.randrange(block_count) * block_size]
    while apple_pos in snake_pos:
        apple_pos = [random.randrange(block_count) * block_size, random.randrange(block_count) * block_size]
    t_score += 1
    return apple_pos, t_score


def display_score(display_text, display_text_2, network):
    large_text = pygame.font.Font('freesansbold.ttf', 35)
    large_text_2 = pygame.font.Font('freesansbold.ttf', 25)
    text_surf = large_text.render(display_text, True, black)
    text_surf_2 = large_text_2.render(display_text_2, True, black)
    text_rect = text_surf.get_rect()
    text_rect_2 = text_surf_2.get_rect()
    text_rect.center = ((display_width / 2), (display_height / 2) - 100)
    text_rect_2.center = ((display_width / 2), (display_height / 2) - 60)
    display.fill(window_color)
    display.blit(text_surf, text_rect)
    display.blit(text_surf_2, text_rect_2)
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game(False, network, "slow")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                game(True, network, "fast")


def is_on_grid(s_head):
    if s_head[0] % block_size == 0 and s_head[1] % block_size == 0:
        return 1
    return 0


def will_be_on_grid(s_head, direction):
    result = 0
    if direction == "left":
        if s_head[0] - snake_speed % block_size == 0:
            result = 1
    elif direction == "right":
        if s_head[0] + snake_speed % block_size == 0:
            result = 1
    elif direction == "up":
        if s_head[1] - snake_speed % block_size == 0:
            result = 1
    elif direction == "down":
        if s_head[1] + snake_speed % block_size == 0:
            result = 1
    return result


def play_game(snake_position, snake_head, apple_position, neural_n, network, speed):
    """ Main game loop """
    crashed = False
    crashed_counter = 2
    prev_button_direction = "right"
    button_direction = "right"
    current_direction = "right"
    score = 0
    draw_grid()
    grid = display.copy()
    counter = 0
    add_counter = 0
    direction_stack = []
    neural_network_input = np.zeros((network.sizes[0], 1))
    moves = [pygame.K_LEFT, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN]
    moves_left = 300
    moves_made = 0
    fitness = 0

    while crashed_counter > 0:
        if neural_n and is_on_grid(snake_head):
            network_move = np.argmax(network.feedforward(neural_network_input))
            cur_dir = SnakeTrainer.calculate_direction_output(current_direction)
            cur_dir = np.argmax(cur_dir)
            # Turn left
            if network_move == 0:
                if cur_dir == 0:
                    cur_dir = 3
                else:
                    cur_dir -= 1
            # Go straight
            elif network_move == 1:
                pass
            # Turn right
            elif network_move == 2:
                if cur_dir == 3:
                    cur_dir = 0
                else:
                    cur_dir += 1
            my_key = moves[cur_dir]
            event = pygame.event.Event(pygame.KEYDOWN, {"key": my_key})
            pygame.event.post(event)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            # This stack thing is for really quick moves
            # If we don't use this sometimes moves can lag
            if len(direction_stack) == 1 and current_direction == button_direction:
                button_direction = direction_stack[0]
                direction_stack.pop()
            else:
                if event.type == pygame.KEYDOWN and current_direction == button_direction:
                    if event.key == pygame.K_LEFT and prev_button_direction != "right":
                        button_direction = "left"
                    elif event.key == pygame.K_RIGHT and prev_button_direction != "left":
                        button_direction = "right"
                    elif event.key == pygame.K_UP and prev_button_direction != "down":
                        button_direction = "up"
                    elif event.key == pygame.K_DOWN and prev_button_direction != "up":
                        button_direction = "down"
                elif event.type == pygame.KEYDOWN and len(direction_stack) == 0:
                    if event.key == pygame.K_LEFT and button_direction != "right":
                        direction_stack.append("left")
                    elif event.key == pygame.K_RIGHT and button_direction != "left":
                        direction_stack.append("right")
                    elif event.key == pygame.K_UP and button_direction != "down":
                        direction_stack.append("up")
                    elif event.key == pygame.K_DOWN and button_direction != "up":
                        direction_stack.append("down")
        prev_button_direction = button_direction
        display.fill(window_color)
        display.blit(grid, (0, 0))
        display_apple(apple_position)

        if snake_head == apple_position:
            apple_position, score = collision_with_apple(score, snake_position)
            moves_left += 120 * score
            fitness += 10
            add_counter = 8
        if add_counter > 0:
            snake_position.insert(0, list(snake_head))
            add_counter -= 1

        distance_to_apple = SnakeTrainer.calculate_real_distance_to_apple(snake_head, apple_position)
        snake_position, apple_position, current_direction = update_snake(snake_head, snake_position, apple_position,
                                                                         button_direction, current_direction)
        new_distance_to_apple = SnakeTrainer.calculate_real_distance_to_apple(snake_head, apple_position)

        if is_on_grid(snake_head):
            if new_distance_to_apple < distance_to_apple:
                fitness += 1
            elif new_distance_to_apple > distance_to_apple:
                fitness -= 1.5

            # I don't exactly know why but this only works with back clear even though we don't need it
            right_clear = SnakeTrainer.is_right_clear(snake_position, current_direction)
            left_clear = SnakeTrainer.is_left_clear(snake_position, current_direction)
            back_clear = SnakeTrainer.is_back_clear(snake_position, current_direction)
            straight_clear = SnakeTrainer.is_straight_clear(snake_position, current_direction)
            food_right = SnakeTrainer.is_food_to_the_right(snake_head, apple_position, current_direction)
            food_left = SnakeTrainer.is_food_to_the_left(snake_head, apple_position, current_direction)
            food_ahead = SnakeTrainer.is_food_straight_ahead(snake_head, apple_position, current_direction)
            # food_behind = SnakeTrainer.is_food_behind(snake_head, apple_position, current_direction)
            neural_network_input = [left_clear, straight_clear, right_clear,
                                    food_right, food_left, food_ahead]
            # self_distances = SnakeTrainer.calculate_distance_to_self(snake_position, display_width, display_height)
            # wall_distances = SnakeTrainer.calculate_distance_to_walls(snake_head, display_width, display_height)
            # apple_distances = SnakeTrainer.calculate_distance_to_apple(snake_head, apple_position)
            # neural_network_input = []
            # for i in apple_distances:
            #     neural_network_input.append(i)
            # for i in wall_distances:
            #     neural_network_input.append(i)
            # for i in self_distances:
            #     neural_network_input.append(i)
            # neural_network_input.append(cur_dir)
            # neural_network_input = np.array([self_distances, wall_distances, apple_distances])
            neural_network_input = np.array([neural_network_input]).reshape((network.sizes[0], 1))

        display_snake(snake_position)
        pygame.display.set_caption("Snake  Skor: " + str(score))
        pygame.display.update()

        if speed == "slow":
            clock.tick(Settings.slow_speed)
        elif speed == "middle":
            clock.tick(Settings.middle_speed)
        elif speed == "fast":
            clock.tick(Settings.fast_speed)

        if counter < 60:
            counter += 1
        else:
            if collision_with_boundaries(snake_head) or collision_with_self(snake_position):
                crashed = True
            elif moves_left <= 0:
                crashed = True
        if crashed:
            crashed_counter -= 1

        moves_left -= 1
        moves_made += 1

    if speed in ("slow", "middle"):
        time.sleep(0.5)

    fitness = fitness + math.sqrt(moves_made)

    return fitness, score


def game(neural, network, speed):
    snake_position_start = []
    for i in range(24):
        snake_position_start.append([display_width / 2 - i * 8, display_height / 2])
    snake_head_start = snake_position_start[0]
    apple_position_start = [random.randrange(block_count) * block_size,
                            random.randrange(block_count) * block_size]
    display.fill(window_color)
    pygame.display.update()

    if neural:
        fitness, final_score = play_game(snake_position_start, snake_head_start,
                                         apple_position_start, neural, network, speed)
    else:
        fitness, final_score = play_game(snake_position_start, snake_head_start,
                                         apple_position_start, neural, network, speed)
        final_text = "Skorunuz: " + str(final_score)
        final_text_2 = "Tekrar Denemek İçin R'ye Basın"
        display_score(final_text, final_text_2, network)
    return fitness


if __name__ == "__main__":
    snake_network = NeuralNetwork.Network(Settings.network_size)
    game(False, snake_network, "slow")
