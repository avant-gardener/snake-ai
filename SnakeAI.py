import pygame
import random
import time
import numpy as np
import NeuralNetwork
import SnakeTrainer


red = (200, 30, 30)
black = (0, 0, 0)
gray = (70, 70, 70)
window_color = (200, 200, 200)
display_width = 512
display_height = 512
block_size = 32
display = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()


def draw_grid():
    for x in range(display_width):
        for y in range(display_height):
            rect = pygame.Rect(x * block_size, y * block_size, block_size, block_size)
            pygame.draw.rect(display, gray, rect, 1)


def display_snake(snake_pos):
    color = (0, 0, 0)
    snake_size_x = block_size - 4
    snake_size_y = block_size - 4
    for position in snake_pos:
        if position == snake_pos[0]:
            pygame.draw.rect(display, color, pygame.Rect(position[0], position[1], snake_size_x + 4, snake_size_y + 4))
        else:
            pygame.draw.rect(display, color, pygame.Rect(position[0] + 2, position[1] + 2, snake_size_x, snake_size_y))


def display_apple(apple_pos):
    pygame.draw.rect(display, red, pygame.Rect(apple_pos[0] + 2, apple_pos[1] + 2, 28, 28))


def update_snake(snake_head_pos, snake_pos, apple_pos, direction, cur_direction):
    if cur_direction != direction:
        if snake_head_pos[0] % block_size == 0 and snake_head_pos[1] % block_size == 0:
            cur_direction = direction
    if cur_direction == "right":
        snake_head_pos[0] += 4
    elif cur_direction == "left":
        snake_head_pos[0] -= 4
    elif cur_direction == "up":
        snake_head_pos[1] -= 4
    elif cur_direction == "down":
        snake_head_pos[1] += 4
    snake_pos.insert(0, list(snake_head_pos))
    snake_pos.pop()

    return snake_pos, apple_pos, cur_direction


def collision_with_boundaries(snake_h):
    if snake_h[0] >= display_height or snake_h[0] < 0 or snake_h[1] >= display_width or snake_h[1] < 0:
        return 1
    else:
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
    apple_pos = [random.randrange(1, 16) * block_size, random.randrange(1, 16) * block_size]
    while apple_pos in snake_pos:
        apple_pos = [random.randrange(1, 16) * block_size, random.randrange(1, 16) * block_size]
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
    display.fill([200, 200, 200])
    display.blit(text_surf, text_rect)
    display.blit(text_surf_2, text_rect_2)
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game(False, network)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                game(True, network)


def is_on_grid(s_head):
    if s_head[0] % block_size == 0 and s_head[1] % block_size == 0:
        return 1
    else:
        return 0


def play_game(snake_position, snake_head, apple_position, neural_n, network):
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
    train_data = []
    move = 1
    neural_network_input = np.zeros((11, 1))
    moves = [pygame.K_LEFT, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN]
    moves_left = 200
    moves_made = 0

    while crashed_counter > 0:
        if neural_n:
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
                    else:
                        button_direction = button_direction
                elif event.type == pygame.KEYDOWN and len(direction_stack) == 0:
                    if event.key == pygame.K_LEFT and button_direction != "right":
                        direction_stack.append("left")
                    elif event.key == pygame.K_RIGHT and button_direction != "left":
                        direction_stack.append("right")
                    elif event.key == pygame.K_UP and button_direction != "down":
                        direction_stack.append("up")
                    elif event.key == pygame.K_DOWN and button_direction != "up":
                        direction_stack.append("down")
                    else:
                        button_direction = button_direction
        if current_direction != button_direction:
            direction_changed = True
        else:
            direction_changed = False
        prev_button_direction = button_direction
        display.fill(window_color)
        display.blit(grid, (0, 0))
        display_apple(apple_position)

        if snake_head == apple_position:
            apple_position, score = collision_with_apple(score, snake_position)
            moves_left += 100 * score
            add_counter = 8
        if add_counter > 0:
            snake_position.insert(0, list(snake_head))
            add_counter -= 1

        but_direction = SnakeTrainer.calculate_direction_input(button_direction)
        cur_direction = SnakeTrainer.calculate_direction_input(current_direction)

        if direction_changed:
            if cur_direction == 3 and but_direction == 0:
                move = 2
            elif cur_direction == 0 and but_direction == 3:
                move = 0
            elif cur_direction < but_direction:
                move = 2
            elif cur_direction > but_direction:
                move = 0
        else:
            move = 1
        neural_network_output = np.zeros((3, 1))
        neural_network_output[move] = 1.0
        if is_on_grid(snake_head) and counter >= 60:
            # I don't exactly know why but this only works with back clear even though I don't need it
            right_clear = SnakeTrainer.is_right_clear(snake_position, current_direction)
            left_clear = SnakeTrainer.is_left_clear(snake_position, current_direction)
            back_clear = SnakeTrainer.is_back_clear(snake_position, current_direction)
            straight_clear = SnakeTrainer.is_straight_clear(snake_position, current_direction)
            apple_distances = SnakeTrainer.calculate_distance_to_apple(snake_head, apple_position)
            neural_network_input = [left_clear, straight_clear, right_clear]
            for i in apple_distances:
                neural_network_input.append(i)
            neural_network_input = np.array([neural_network_input])
            neural_network_input = neural_network_input[0].reshape((11, 1))
            if not neural_n:
                train_data.append((neural_network_input, neural_network_output))

        snake_position, apple_position, current_direction = update_snake(snake_head, snake_position, apple_position,
                                                                         button_direction, current_direction)

        display_snake(snake_position)
        pygame.display.set_caption("Snake  Skor: " + str(score))
        pygame.display.update()
        if not neural_n:
            clock.tick(60)
        else:
            clock.tick(240)

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

    time.sleep(0.5)

    return moves_made, train_data, score


def game(neural, network):
    snake_position_start = []
    for i in range(24):
        snake_position_start.append([display_width / 2 - i * 8, display_height / 2])
    snake_head_start = snake_position_start[0]
    apple_position_start = [random.randrange(1, 16) * block_size, random.randrange(1, 16) * block_size]
    display.fill(window_color)
    pygame.display.update()

    if neural:
        total_moves, training_data, final_score = play_game(snake_position_start, snake_head_start, apple_position_start
                                                            , neural, network)
        return total_moves
    else:
        total_moves, training_data, final_score = play_game(snake_position_start, snake_head_start, apple_position_start
                                                            , neural, network)
        snake_network.sgd(training_data, 10, 10, 1, training_data)
        final_text = "Skorunuz: " + str(final_score)
        final_text_2 = "Tekrar Denemek İçin R'ye Basın"
        display_score(final_text, final_text_2, network)


if __name__ == "__main__":
    pygame.init()

    snake_network = NeuralNetwork.Network([11, 10, 3])
    game_number = 0
    game(False, snake_network)
