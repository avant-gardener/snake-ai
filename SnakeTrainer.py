import numpy as np
import math
import Settings

block_size = Settings.block_size
display_height = Settings.display_height
display_width = Settings.display_width


def calculate_distance_to_self(snake_position, width, height):
    distances = np.zeros((8, ))
    # Distances are indexed starting from south and ending in south-east direction
    # ex: distances[4] == distance_to_up
    snake_head = snake_position[0]
    for snake_part in snake_position:
        if snake_part[0] - snake_head[0] == snake_part[1] - snake_head[1]:
            if snake_part[0] - snake_head[0] < 0:
                # Up-Left
                distances[3] = math.sqrt(2 * ((snake_part[0] - snake_head[0])**2))
            else:
                # Down-Right
                distances[7] = math.sqrt(2 * ((snake_part[0] - snake_head[0])**2))
        elif snake_part[0] - snake_head[0] == -1 * (snake_part[1] - snake_head[1]):
            if snake_part[0] - snake_head[0] < 0:
                # Down-Left
                distances[1] = math.sqrt(2 * ((snake_part[0] - snake_head[0])**2))
            else:
                # Up-Right
                distances[5] = math.sqrt(2 * ((snake_part[0] - snake_head[0])**2))
        elif snake_part[0] == snake_head[0]:
            if snake_part[1] > snake_head[1]:
                # Down
                distances[0] = abs(snake_part[1] - snake_head[1] + block_size)
            else:
                # Up
                distances[4] = abs(snake_part[1] - snake_head[1] + block_size)
        elif snake_part[1] == snake_head[1]:
            if snake_part[0] > snake_head[0]:
                # Right
                distances[6] = abs(snake_part[0] - snake_head[0] + block_size)
            else:
                # Left
                distances[2] = abs(snake_part[0] - snake_head[0] + block_size)

    for i in range(len(distances)):
        if distances[i] == 0:
            if i == 0:
                distances[i] = height - snake_head[1]
            elif i == 2:
                distances[i] = snake_head[0]
            elif i == 4:
                distances[i] = snake_head[1]
            elif i == 6:
                distances[i] = width - snake_head[0]
            elif i == 1 or 3:
                if snake_head[0] < snake_head[1]:
                    distances[i] = math.sqrt(2 * (snake_head[0] ** 2))
                else:
                    distances[i] = math.sqrt(2 * (snake_head[1] ** 2))
            elif i == 5 or 7:
                if snake_head[0] < height - snake_head[1]:
                    distances[i] = math.sqrt(2 * ((height - snake_head[1]) ** 2))
                else:
                    distances[i] = math.sqrt(2 * ((width - snake_head[0]) ** 2))

    distances = rescale_data(distances, 512)

    return distances


def calculate_distance_to_apple(snake_head, apple_position):
    distances = np.zeros((8, ))

    if apple_position[0] - snake_head[0] == apple_position[1] - snake_head[1]:
        if apple_position[0] - snake_head[0] < 0:
            # Up-Left
            distances[3] = math.sqrt(2 * ((apple_position[0] - snake_head[0]) ** 2))
            distances[3] = 1
        else:
            # Down-Right
            distances[7] = math.sqrt(2 * ((apple_position[0] - snake_head[0]) ** 2))
            distances[7] = 1
    elif apple_position[0] - snake_head[0] == -1 * (apple_position[1] - snake_head[1]):
        if apple_position[0] - snake_head[0] < 0:
            # Down-Left
            distances[1] = math.sqrt(2 * ((apple_position[0] - snake_head[0]) ** 2))
            distances[1] = 1
        else:
            # Up-Right
            distances[5] = math.sqrt(2 * ((apple_position[0] - snake_head[0]) ** 2))
            distances[5] = 1
    elif apple_position[0] == snake_head[0]:
        if apple_position[1] > snake_head[1]:
            # Down
            distances[0] = abs(apple_position[1] - snake_head[1] + block_size)
            distances[0] = 1
        else:
            # Up
            distances[4] = abs(apple_position[1] - snake_head[1] + block_size)
            distances[4] = 1
    elif apple_position[1] == snake_head[1]:
        if apple_position[0] > snake_head[0]:
            # Right
            distances[6] = abs(apple_position[0] - snake_head[0] + block_size)
            distances[6] = 1
        else:
            # Left
            distances[2] = abs(apple_position[0] - snake_head[0] + block_size)
            distances[2] = 1

    # distances = rescale_data(distances, 512)

    return distances


def calculate_distance_to_walls(snake_head, width, height):
    distances = np.zeros((8, ))

    distances[0] = height - snake_head[1]
    distances[2] = snake_head[0]
    distances[4] = snake_head[1]
    distances[6] = width - snake_head[0]
    if snake_head[0] < snake_head[1]:
        distances[1] = math.sqrt(2 * (snake_head[0] ** 2))
        distances[3] = math.sqrt(2 * (snake_head[0] ** 2))
    else:
        distances[1] = math.sqrt(2 * (snake_head[1] ** 2))
        distances[3] = math.sqrt(2 * (snake_head[1] ** 2))
    if snake_head[0] < height - snake_head[1]:
        distances[5] = math.sqrt(2 * ((height - snake_head[1]) ** 2))
        distances[7] = math.sqrt(2 * ((height - snake_head[1]) ** 2))
    else:
        distances[5] = math.sqrt(2 * ((width - snake_head[0]) ** 2))
        distances[7] = math.sqrt(2 * ((width - snake_head[0]) ** 2))

    distances = rescale_data(distances, 512)

    return distances


def calculate_direction_output(current_direction):
    if current_direction == "left":
        nn_input = 0
    elif current_direction == "up":
        nn_input = 1
    elif current_direction == "right":
        nn_input = 2
    elif current_direction == "down":
        nn_input = 3
    else:
        nn_input = 0
    output = np.zeros((4, 1))
    output[nn_input] = 1.0

    return output


def calculate_direction_input(current_direction):
    if current_direction == "left":
        nn_input = 0
    elif current_direction == "up":
        nn_input = 1
    elif current_direction == "right":
        nn_input = 2
    elif current_direction == "down":
        nn_input = 3
    else:
        nn_input = 0

    return nn_input


def is_left_clear(snake_position, current_direction):
    snake_head = snake_position[0]
    if current_direction == "left":
        snake_position[0][1] += block_size
    elif current_direction == "up":
        snake_position[0][0] -= block_size
    elif current_direction == "right":
        snake_position[0][1] -= block_size
    elif current_direction == "down":
        snake_position[0][0] += block_size
    if collision_with_self(snake_position) or collision_with_boundaries(snake_head):
        snake_position[0] = snake_head
        return 0
    else:
        snake_position[0] = snake_head
        return 1


def is_straight_clear(snake_position, current_direction):
    snake_head = snake_position[0]
    if current_direction == "left":
        snake_position[0][0] -= block_size
    elif current_direction == "up":
        snake_position[0][1] -= block_size
    elif current_direction == "right":
        snake_position[0][0] += block_size
    elif current_direction == "down":
        snake_position[0][1] += block_size
    if collision_with_self(snake_position) or collision_with_boundaries(snake_head):
        return 0
    else:
        return 1


def is_right_clear(snake_position, current_direction):
    snake_head = snake_position[0]
    if current_direction == "left":
        snake_position[0][1] -= block_size
    elif current_direction == "up":
        snake_position[0][0] += block_size
    elif current_direction == "right":
        snake_position[0][1] += block_size
    elif current_direction == "down":
        snake_position[0][0] -= block_size
    if collision_with_self(snake_position) or collision_with_boundaries(snake_head):
        return 0
    else:
        return 1


def is_back_clear(snake_position, current_direction):
    snake_head = snake_position[0]
    if current_direction == "left":
        snake_position[0][0] += block_size
    elif current_direction == "up":
        snake_position[0][1] += block_size
    elif current_direction == "right":
        snake_position[0][0] -= block_size
    elif current_direction == "down":
        snake_position[0][1] -= block_size
    if collision_with_self(snake_position) or collision_with_boundaries(snake_head):
        return 0
    else:
        return 1


def rescale_data(data, factor):
    for i in range(len(data)):
        data[i] = data[i] / factor
    return data


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


def collision_with_boundaries(snake_h):
    if snake_h[0] >= display_height or snake_h[0] < 0 or snake_h[1] >= display_width or snake_h[1] < 0:
        return 1
    else:
        return 0


def calculate_real_distance_to_apple(snake_head, apple_position):
    x = abs(snake_head[0] - apple_position[0])
    y = abs(snake_head[1] - apple_position[1])
    return math.sqrt(x**2 + y**2)


def is_food_to_the_right(snake_head, apple_position, direction):
    result = 0
    if direction == "left" and snake_head[1] > apple_position[1]:
        result = 1
    elif direction == "up" and snake_head[0] < apple_position[0]:
        result = 1
    elif direction == "right" and snake_head[1] < apple_position[1]:
        result = 1
    elif direction == "down" and snake_head[0] > apple_position[0]:
        result = 1
    return result


def is_food_to_the_left(snake_head, apple_position, direction):
    result = 1
    if direction == "left" and snake_head[1] > apple_position[1]:
        result = 0
    elif direction == "up" and snake_head[0] < apple_position[0]:
        result = 0
    elif direction == "right" and snake_head[1] < apple_position[1]:
        result = 0
    elif direction == "down" and snake_head[0] > apple_position[0]:
        result = 0
    return result


def is_food_straight_ahead(snake_head, apple_position, direction):
    result = 0
    if direction == "left" and snake_head[0] > apple_position[0]:
        result = 1
    elif direction == "up" and snake_head[1] > apple_position[1]:
        result = 1
    elif direction == "right" and snake_head[0] < apple_position[0]:
        result = 1
    elif direction == "down" and snake_head[1] < apple_position[1]:
        result = 1
    return result
