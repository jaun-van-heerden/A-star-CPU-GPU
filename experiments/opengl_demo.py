import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *

# Initialize Pygame
pygame.init()

# Window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# OpenGL setup
glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Simple 3D Rendering")

# Camera settings
gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glRotatef(1, 3, 1, 1)  # Rotate the cube

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBegin(GL_QUADS)

    # Define vertices and faces for a simple cube
    vertices = [
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, 1]
    ]

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]

    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()
    pygame.display.flip()
    pygame.time.wait(10)
