#version 330 core
in vec4 FragColor;
out vec4 FragColor;

uniform float transparency;

void main()
{
    FragColor = vec4(1.0, 0.0, 0.0, transparency); // Red color with transparency
}
